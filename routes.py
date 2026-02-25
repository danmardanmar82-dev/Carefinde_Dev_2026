import hashlib
import os
import secrets
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from urllib.parse import urlparse
import stripe
from fastapi import FastAPI, APIRouter, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from pydantic import BaseModel

# ─── Config ──────────────────────────────────────────────────────────────────
SESSION_COOKIE = "cf_session"
SESSIONS: dict = {}


def get_db_url() -> str:
    raw = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
    if not raw:
        raise RuntimeError("DATABASE_URL nie jest ustawiony. Dodaj zmienną środowiskową NEON_DATABASE_URL.")
    # strip whitespace/comments and remove unsupported channel_binding param
    return raw.strip().split()[0].replace("&channel_binding=require", "")


def get_openai():
    key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_OPENAI_API_KEY")
    )
    return OpenAI(api_key=key) if key else None


def get_stripe_secret():
    return (
        os.environ.get("STRIPE_SECRET_KEY")
        or os.environ.get("STRIPE_STRIPE_SECRET_KEY")
        or ""
    )


def get_stripe_pub():
    return (
        os.environ.get("STRIPE_PUBLISHABLE_KEY")
        or os.environ.get("STRIPE_STRIPE_PUBLISHABLE_KEY")
        or ""
    )


# ─── DB helpers ──────────────────────────────────────────────────────────────
@contextmanager
def get_db():
    """Context manager zwracający połączenie psycopg2 z RealDictCursor."""
    p = urlparse(get_db_url())
    conn = psycopg2.connect(
        host=p.hostname,
        port=p.port or 5432,
        dbname=p.path.lstrip("/"),
        user=p.username,
        password=p.password,
        sslmode="require",
        connect_timeout=10,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_expired_premium():
    today = datetime.now().date()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE companies SET is_premium=FALSE WHERE premium_expiry < %s AND is_premium=TRUE",
                (today,),
            )


def ensure_tables():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT DEFAULT 'partner'
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT,
                    city TEXT,
                    lat DOUBLE PRECISION,
                    lng DOUBLE PRECISION,
                    is_premium BOOLEAN DEFAULT FALSE,
                    premium_expiry DATE,
                    contact_info TEXT,
                    total_slots INTEGER DEFAULT 0,
                    occupied_slots INTEGER DEFAULT 0,
                    ad_text TEXT,
                    owner_username TEXT,
                    is_real BOOLEAN DEFAULT TRUE,
                    ppc_budget NUMERIC(10,2) DEFAULT 0
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS contact_messages (
                    id SERIAL PRIMARY KEY,
                    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
                    sender_name TEXT,
                    sender_email TEXT,
                    message TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            for col_sql in [
                "ALTER TABLE companies ADD COLUMN IF NOT EXISTS ppc_budget NUMERIC(10,2) DEFAULT 0",
                "ALTER TABLE companies ADD COLUMN IF NOT EXISTS has_vacancy BOOLEAN DEFAULT FALSE",
                "ALTER TABLE companies ADD COLUMN IF NOT EXISTS vacancy_text TEXT",
                "ALTER TABLE companies ADD COLUMN IF NOT EXISTS photo_url TEXT",
                "ALTER TABLE companies ADD COLUMN IF NOT EXISTS video_url TEXT",
                "ALTER TABLE companies ADD COLUMN IF NOT EXISTS contact_email TEXT",
            ]:
                cur.execute(col_sql)
            cur.execute("""
                INSERT INTO users (username, password, role)
                VALUES ('admin', 'admin123', 'admin')
                ON CONFLICT (username) DO NOTHING
            """)


# ─── Session helpers ──────────────────────────────────────────────────────────
def create_session(user_id: int, username: str, role: str) -> str:
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = {"user_id": user_id, "username": username, "role": role}
    return token


def get_session(request: Request) -> Optional[dict]:
    token = request.cookies.get(SESSION_COOKIE)
    return SESSIONS.get(token) if token else None


def require_session(request: Request) -> dict:
    sess = get_session(request)
    if not sess:
        raise HTTPException(status_code=401, detail="Niet ingelogd")
    return sess


def require_admin(request: Request) -> dict:
    sess = require_session(request)
    if sess.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Geen toegang")
    return sess


# ─── Pydantic models ─────────────────────────────────────────────────────────
class LoginData(BaseModel):
    username: str
    password: str


class RegisterData(BaseModel):
    username: str
    password: str


class AIMessage(BaseModel):
    message: str
    context: Optional[str] = None


class FirmData(BaseModel):
    name: str
    cat: str
    city: Optional[str] = None
    lat: float
    lng: float
    contact: Optional[str] = None
    contact_email: Optional[str] = None
    total_slots: Optional[int] = 0
    occupied_slots: Optional[int] = 0
    ad_text: Optional[str] = None
    photo_url: Optional[str] = None
    video_url: Optional[str] = None


class VacancyData(BaseModel):
    has_vacancy: bool
    vacancy_text: Optional[str] = None


class ContactMessage(BaseModel):
    company_id: int
    sender_name: str
    sender_email: str
    message: str


class AdminFirmUpdate(BaseModel):
    company_id: int
    is_premium: Optional[int] = None
    premium_expiry: Optional[str] = None


# ─── File hash ────────────────────────────────────────────────────────────────
def get_file_hash(filepath: str) -> str:
    try:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except Exception:
        return "0"


# ─── App factory ─────────────────────────────────────────────────────────────
def create_app(static_dir: str) -> FastAPI:
    ensure_tables()

    app = FastAPI(title="Carefinder NL")
    api = APIRouter()
    templates = Jinja2Templates(directory=static_dir)

    # ── Pages ─────────────────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        css_hash = get_file_hash(os.path.join(static_dir, "styles.css"))
        js_hash = get_file_hash(os.path.join(static_dir, "app.js"))
        return templates.TemplateResponse(
            request, "index.html",
            {"css_hash": css_hash, "js_hash": js_hash, "stripe_pub": get_stripe_pub()},
        )

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        return templates.TemplateResponse(request, "login.html", {})

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_page(request: Request):
        sess = get_session(request)
        if not sess:
            return RedirectResponse("/login")
        tpl = "dashboard_admin.html" if sess["role"] == "admin" else "dashboard_partner.html"
        return templates.TemplateResponse(request, tpl, {"username": sess["username"]})

    @app.get("/algemene-voorwaarden", response_class=HTMLResponse)
    def algemene_voorwaarden():
        with open(Path(static_dir) / "algemene_voorwaarden.html", encoding="utf-8") as f:
            return f.read()

    @app.get("/logout")
    def logout(request: Request):
        token = request.cookies.get(SESSION_COOKIE)
        if token and token in SESSIONS:
            del SESSIONS[token]
        resp = RedirectResponse("/login", status_code=302)
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    # ── Auth API ──────────────────────────────────────────────────────────────
    @api.post("/login")
    def login_api(data: LoginData):
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM users WHERE username=%s AND password=%s",
                    (data.username, data.password),
                )
                user = cur.fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="Foutieve inloggegevens")
        token = create_session(user["id"], user["username"], user["role"])
        resp = JSONResponse({"status": "success", "role": user["role"]})
        resp.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="none", secure=True, max_age=86400 * 7)
        return resp

    @api.post("/register")
    def register_api(data: RegisterData):
        if len(data.username) < 3 or len(data.password) < 6:
            raise HTTPException(status_code=400, detail="Gebruikersnaam min 3, wachtwoord min 6 tekens")
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO users (username, password, role) VALUES (%s, %s, 'partner')",
                        (data.username, data.password),
                    )
        except psycopg2.errors.UniqueViolation:
            raise HTTPException(status_code=409, detail="Gebruikersnaam al in gebruik")
        return {"status": "success"}

    @api.get("/me")
    def me(request: Request):
        sess = get_session(request)
        if not sess:
            return {"logged_in": False}
        return {"logged_in": True, "username": sess["username"], "role": sess["role"]}

    # ── Firms API ─────────────────────────────────────────────────────────────
    @api.get("/firms")
    def get_firms(
        category: Optional[str] = None,
        city: Optional[str] = None,
        premium_only: Optional[bool] = False,
        limit: int = 500,
        offset: int = 0,
    ):
        check_expired_premium()
        where, params = [], []
        if category:
            where.append("category=%s"); params.append(category)
        if city:
            where.append("city ILIKE %s"); params.append(f"%{city}%")
        if premium_only:
            where.append("is_premium=TRUE")
        sql = "SELECT id,name,lat,lng,category,city,is_premium,contact_info,contact_email,total_slots,occupied_slots,ad_text,has_vacancy,vacancy_text,photo_url,video_url FROM companies"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY is_premium DESC, name ASC LIMIT %s OFFSET %s"
        params += [limit, offset]
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    @api.get("/firms/stats")
    def firms_stats():
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as total FROM companies"); total = cur.fetchone()["total"]
                cur.execute("SELECT COUNT(*) as n FROM companies WHERE category='thuiszorg'"); thuiszorg = cur.fetchone()["n"]
                cur.execute("SELECT COUNT(*) as n FROM companies WHERE category='verzorgingshuis'"); verzorgingshuis = cur.fetchone()["n"]
                cur.execute("SELECT COUNT(*) as n FROM companies WHERE is_premium=TRUE"); premium = cur.fetchone()["n"]
        return {"total": total, "thuiszorg": thuiszorg, "verzorgingshuis": verzorgingshuis, "premium": premium}

    @api.get("/firms/search")
    def search_firms(q: str, limit: int = 20):
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id,name,lat,lng,category,city,is_premium,contact_info FROM companies WHERE name ILIKE %s OR city ILIKE %s ORDER BY is_premium DESC LIMIT %s",
                    (f"%{q}%", f"%{q}%", limit),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    # ── Partner API ───────────────────────────────────────────────────────────
    @api.post("/partner/update")
    def partner_update(data: FirmData, request: Request):
        sess = require_session(request)
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM companies WHERE owner_username=%s", (sess["username"],))
                existing = cur.fetchone()
                if existing:
                    cur.execute(
                        """UPDATE companies SET name=%s,category=%s,city=%s,lat=%s,lng=%s,
                           contact_info=%s,contact_email=%s,total_slots=%s,occupied_slots=%s,
                           ad_text=%s,photo_url=%s,video_url=%s WHERE owner_username=%s""",
                        (data.name, data.cat, data.city, data.lat, data.lng, data.contact,
                         data.contact_email, data.total_slots, data.occupied_slots,
                         data.ad_text, data.photo_url, data.video_url, sess["username"]),
                    )
                else:
                    cur.execute(
                        """INSERT INTO companies (name,category,city,lat,lng,contact_info,contact_email,
                           total_slots,occupied_slots,ad_text,photo_url,video_url,owner_username,is_premium,is_real)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FALSE,TRUE)""",
                        (data.name, data.cat, data.city, data.lat, data.lng, data.contact,
                         data.contact_email, data.total_slots, data.occupied_slots,
                         data.ad_text, data.photo_url, data.video_url, sess["username"]),
                    )
        return {"status": "success"}

    @api.post("/partner/vacancy")
    def partner_vacancy(data: VacancyData, request: Request):
        sess = require_session(request)
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE companies SET has_vacancy=%s, vacancy_text=%s WHERE owner_username=%s",
                    (data.has_vacancy, data.vacancy_text, sess["username"]),
                )
        return {"status": "success"}

    @api.post("/contact-message")
    def send_contact_message(data: ContactMessage):
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM companies WHERE id=%s", (data.company_id,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Bedrijf niet gevonden")
                cur.execute(
                    "INSERT INTO contact_messages (company_id, sender_name, sender_email, message) VALUES (%s,%s,%s,%s)",
                    (data.company_id, data.sender_name, data.sender_email, data.message),
                )
        return {"status": "success"}

    @api.get("/partner/messages")
    def partner_messages(request: Request):
        sess = require_session(request)
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM companies WHERE owner_username=%s", (sess["username"],))
                row = cur.fetchone()
                if not row:
                    return []
                cur.execute(
                    "SELECT id, sender_name, sender_email, message, created_at FROM contact_messages WHERE company_id=%s ORDER BY created_at DESC LIMIT 50",
                    (row["id"],),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    @api.get("/firms/nearby")
    def firms_nearby(lat: float, lng: float, radius_km: float = 50, limit: int = 200):
        check_expired_premium()
        sql = """
            SELECT id, name, lat, lng, category, city, is_premium, contact_info, contact_email,
                   total_slots, occupied_slots, ad_text, has_vacancy, vacancy_text, photo_url, video_url,
                   (6371 * acos(
                       LEAST(1, cos(radians(%s)) * cos(radians(lat)) *
                       cos(radians(lng) - radians(%s)) +
                       sin(radians(%s)) * sin(radians(lat)))
                   )) AS distance_km
            FROM companies
            WHERE lat IS NOT NULL AND lng IS NOT NULL
            HAVING (6371 * acos(
                       LEAST(1, cos(radians(%s)) * cos(radians(lat)) *
                       cos(radians(lng) - radians(%s)) +
                       sin(radians(%s)) * sin(radians(lat)))
                   )) <= %s
            ORDER BY is_premium DESC, distance_km ASC
            LIMIT %s
        """
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (lat, lng, lat, lat, lng, lat, radius_km, limit))
                rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["distance_km"] = round(float(d["distance_km"]), 1)
            result.append(d)
        return result

    @api.get("/partner/firm")
    def get_partner_firm(request: Request):
        sess = require_session(request)
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM companies WHERE owner_username=%s", (sess["username"],))
                row = cur.fetchone()
        return dict(row) if row else {}

    @api.get("/partner/status")
    def partner_status(request: Request):
        sess = require_session(request)
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT is_premium, premium_expiry, ppc_budget FROM companies WHERE owner_username=%s", (sess["username"],)
                )
                row = cur.fetchone()
        if not row:
            return {"is_premium": False, "premium_expiry": None, "ppc_budget": 0}
        expiry = row["premium_expiry"]
        return {
            "is_premium": bool(row["is_premium"]),
            "premium_expiry": str(expiry) if expiry else None,
            "ppc_budget": float(row["ppc_budget"] or 0),
        }

    # ── Admin API ─────────────────────────────────────────────────────────────
    @api.get("/admin/firms")
    def admin_get_firms(request: Request, limit: int = 100, offset: int = 0):
        require_admin(request)
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id,name,category,city,is_premium,premium_expiry,owner_username,contact_info FROM companies ORDER BY is_premium DESC, name ASC LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                rows = cur.fetchall()
                cur.execute("SELECT COUNT(*) as total FROM companies")
                total = cur.fetchone()["total"]
        firms = []
        for r in rows:
            d = dict(r)
            if d.get("premium_expiry"):
                d["premium_expiry"] = str(d["premium_expiry"])
            firms.append(d)
        return {"firms": firms, "total": total}

    @api.post("/admin/update_firm")
    def admin_update_firm(data: AdminFirmUpdate, request: Request):
        require_admin(request)
        if data.is_premium is not None:
            expiry = data.premium_expiry or (
                (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d") if data.is_premium else None
            )
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE companies SET is_premium=%s, premium_expiry=%s WHERE id=%s",
                        (bool(data.is_premium), expiry, data.company_id),
                    )
        return {"status": "success"}

    @api.post("/admin/reset-premium")
    def admin_reset_premium(request: Request):
        require_admin(request)
        from datetime import datetime, timedelta
        expiry = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE companies SET is_premium=FALSE, premium_expiry=NULL WHERE is_premium=TRUE")
                count = cur.rowcount
                cur.execute(
                    "UPDATE companies SET is_premium=TRUE, premium_expiry=%s WHERE name='Zesta Zorggroep' AND city='Heerlen'",
                    (expiry,)
                )
        return {"reset": count, "message": f"{count} bedrijven gereset. Alleen Zesta Zorggroep is premium."}

    @api.get("/admin/users")
    def admin_get_users(request: Request):
        require_admin(request)
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, username, role FROM users ORDER BY role, username")
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    # ── AI Assistant (Gemini) ─────────────────────────────────────────────────
    SYSTEM_PROMPT = """Je bent een gespecialiseerde AI-assistent voor Carefinder NL — het toonaangevende platform voor thuiszorg en verzorgingshuizen in heel Nederland.

Je hebt diepgaande expertise in:
🏥 THUISZORG: wijkverpleging (ZVW), persoonlijke verzorging, begeleiding thuis, huishoudelijke hulp, respijtzorg, mantelzorgondersteuning, bejaardentehuis, bejaardenhuizen
🏠 VERPLEEG- EN VERZORGINGSHUIZEN: woonzorgcentra, verpleeghuizen, bejaardentehuis, dementiezorg (ELV), revalidatie, GRZ
📋 WETGEVING & FINANCIERING: WLZ (Wet langdurige zorg), WMO (Wet maatschappelijke ondersteuning), ZVW (Zorgverzekeringswet), AWBZ-erfenis
💰 PGB vs. ZIN: persoonsgebonden budget, zorg in natura, zorgkantoor, budgethouder
🔍 INDICATIESTELLING: CIZ-indicatie, ZZP-profielen (1-10), bezwaar & beroep, spoedprocedure
⏳ WACHTLIJSTEN: urgentie, spoedzorg, crisisopvang, wachttijden per regio
🗺️ REGIO'S: alle 12 provincies van Nederland — Noord-Holland, Zuid-Holland, Utrecht, Noord-Brabant, Gelderland, Overijssel, Friesland, Groningen, Drenthe, Zeeland, Limburg, Flevoland
🔬 KWALITEIT: IGJ-toezicht, Kwaliteitskader Verpleeghuiszorg, cliëntervaringsonderzoek
👶 SPECIALISATIES: dementiezorg, palliatieve zorg, GGZ, VG-zorg, Niet-aangeboren Hersenletsel

KAARTINSTRUCTIES: Wanneer je aanbieders noemt of zoekt, geef dan ook aan of de kaart moet navigeren.
Als de gebruiker vraagt naar aanbieders in de buurt, gebruik dan de beschikbare aanbieders in de context en noem ze bij naam.
De kaart dekt heel Nederland — van Groningen tot Zeeland.

Geef altijd concrete, empathische en praktische adviezen.
Communiceer in de taal van de gebruiker: Nederlands (voorkeur), Engels, Pools of Duits.

⛔ STRIKT VERBODEN — beantwoord NOOIT vragen over:
- Financiering, leningen, kredieten, hypotheken of bancaire producten
- Subsidies anders dan directe zorgvergoedingen (WLZ/WMO/ZVW)
- Investeringsadvies of financiële planning
- Verzekeringen anders dan zorgverzekeringen
Als een gebruiker vraagt over financiering of leningen, antwoord dan: "Hiervoor kunt u contact opnemen met uw bank of een financieel adviseur. Ik help alleen bij vragen over thuiszorg en verzorgingshuizen." """

    @api.post("/ai-assistant")
    async def ai_assistant(data: AIMessage):
        api_key = (
            os.environ.get("GEMINI_GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_GEMINI_API_KEY")
        )
        if not api_key:
            return {"reply": "AI-assistent is tijdelijk niet beschikbaar. API-sleutel ontbreekt."}
        try:
            from google import genai as genai_sdk
            client = genai_sdk.Client(api_key=api_key)

            # ── Pobierz lokalizację użytkownika z kontekstu mapy ──────────────
            user_lat, user_lng = None, None
            nearby_firms_list = []
            if data.context:
                import re
                m = re.search(r'Kaartcentrum:\s*([\d.]+),([\d.]+)', data.context)
                if m:
                    try:
                        user_lat = float(m.group(1))
                        user_lng = float(m.group(2))
                    except Exception:
                        pass

            # ── Szukaj firm w promieniu 50 km ─────────────────────────────────
            if user_lat and user_lng:
                try:
                    with get_db() as conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT id, name, lat, lng, category, city, is_premium,
                                       has_vacancy, vacancy_text,
                                       (6371 * acos(
                                           LEAST(1, cos(radians(%s)) * cos(radians(lat)) *
                                           cos(radians(lng) - radians(%s)) +
                                           sin(radians(%s)) * sin(radians(lat)))
                                       )) AS dist
                                FROM companies
                                WHERE lat IS NOT NULL AND lng IS NOT NULL
                                HAVING (6371 * acos(
                                           LEAST(1, cos(radians(%s)) * cos(radians(lat)) *
                                           cos(radians(lng) - radians(%s)) +
                                           sin(radians(%s)) * sin(radians(lat)))
                                       )) <= 50
                                ORDER BY is_premium DESC, dist ASC
                                LIMIT 20
                            """, (user_lat, user_lng, user_lat, user_lat, user_lng, user_lat))
                            rows = cur.fetchall()
                    nearby_firms_list = [dict(r) for r in rows]
                except Exception:
                    pass

            # ── Buduj prompt ──────────────────────────────────────────────────
            prompt = SYSTEM_PROMPT
            if data.context:
                prompt += f"\n\nHuidige kaartcontext: {data.context}"
            if nearby_firms_list:
                firms_desc = "; ".join(
                    f"{r['name']} ({r['category']}, {r.get('city','?')}{' ⭐' if r.get('is_premium') else ''}{' 🔔vacature' if r.get('has_vacancy') else ''})"
                    for r in nearby_firms_list
                )
                prompt += f"\n\nAanbieders binnen 50 km van gebruiker: {firms_desc}"
            prompt += f"\n\nGebruikersvraag: {data.message}"

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            reply = response.text

            # ── Nawigacja mapy ────────────────────────────────────────────────
            map_lat, map_lng, map_label, map_zoom = None, None, None, 13
            map_firms = []
            try:
                with get_db() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT name, lat, lng, city FROM companies WHERE name ILIKE %s AND lat IS NOT NULL ORDER BY is_premium DESC LIMIT 1",
                            (f"%{data.message}%",)
                        )
                        row = cur.fetchone()
                        if row:
                            map_lat, map_lng, map_label, map_zoom = row["lat"], row["lng"], row["name"], 15
                        else:
                            cur.execute(
                                "SELECT name, lat, lng, city FROM companies WHERE city ILIKE %s AND lat IS NOT NULL ORDER BY is_premium DESC LIMIT 1",
                                (f"%{data.message}%",)
                            )
                            row = cur.fetchone()
                            if row:
                                map_lat, map_lng, map_label, map_zoom = row["lat"], row["lng"], row["city"], 12
            except Exception:
                pass

            # ── Pokaż pobliskie firmy na mapie gdy pytanie o "buurt" ──────────
            nearby_keywords = ["buurt", "omgeving", "nearby", "dichtbij", "dichtstbijzijnde", "in de buurt", "naast mij"]
            if any(kw in data.message.lower() for kw in nearby_keywords) and nearby_firms_list:
                map_firms = [
                    {"id": r["id"], "name": r["name"], "lat": r["lat"], "lng": r["lng"],
                     "category": r["category"], "city": r.get("city",""), "is_premium": r.get("is_premium", False),
                     "has_vacancy": r.get("has_vacancy", False), "distance_km": round(float(r.get("dist",0)), 1)}
                    for r in nearby_firms_list if r.get("lat") and r.get("lng")
                ]
                if not map_lat and map_firms:
                    map_lat = user_lat
                    map_lng = user_lng
                    map_zoom = 11
                    map_label = "Aanbieders in uw buurt"

            result = {"reply": reply}
            if map_lat and map_lng:
                result["map_lat"] = map_lat
                result["map_lng"] = map_lng
                result["map_zoom"] = map_zoom
                result["map_label"] = map_label
            if map_firms:
                result["map_firms"] = map_firms
            return result
        except Exception as e:
            return {"reply": f"AI tijdelijk niet beschikbaar. ({str(e)[:120]})"}

    # ── Stripe ────────────────────────────────────────────────────────────────
    @api.post("/create-checkout-session")
    def create_checkout(request: Request):
        sess = require_session(request)
        sk = get_stripe_secret()
        if not sk:
            raise HTTPException(status_code=500, detail="Stripe niet geconfigureerd")
        stripe.api_key = sk
        host = str(request.base_url).rstrip("/")
        try:
            checkout = stripe.checkout.Session.create(
                payment_method_types=["card", "ideal"],
                client_reference_id=sess["username"],
                line_items=[{"price_data": {"currency": "eur", "product_data": {"name": "Carefinder Premium 30 dagen"}, "unit_amount": 15000}, "quantity": 1}],
                mode="payment",
                success_url=f"{host}/payment-success?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{host}/?cancel=true",
            )
            return {"id": checkout.id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @api.post("/firms/{firm_id}/click")
    def track_click(firm_id: int):
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE companies SET ppc_budget = ppc_budget - 1
                       WHERE id = %s AND ppc_budget >= 1""",
                    (firm_id,),
                )
        return {"ok": True}

    @api.post("/create-ppc-topup-session")
    def create_ppc_topup(request: Request):
        sess = require_session(request)
        sk = get_stripe_secret()
        if not sk:
            raise HTTPException(status_code=500, detail="Stripe niet geconfigureerd")
        stripe.api_key = sk
        host = str(request.base_url).rstrip("/")
        try:
            checkout = stripe.checkout.Session.create(
                payment_method_types=["card", "ideal"],
                client_reference_id=sess["username"],
                line_items=[{"price_data": {"currency": "eur", "product_data": {"name": "Carefinder Pay Per Click — €50 tegoed"}, "unit_amount": 5000}, "quantity": 1}],
                mode="payment",
                success_url=f"{host}/payment-ppc-success?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{host}/dashboard?cancel=true",
            )
            return {"id": checkout.id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/payment-ppc-success")
    def payment_ppc_success(session_id: str):
        sk = get_stripe_secret()
        if sk:
            stripe.api_key = sk
            try:
                stripe_sess = stripe.checkout.Session.retrieve(session_id)
                if stripe_sess.payment_status == "paid":
                    with get_db() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE companies SET ppc_budget = COALESCE(ppc_budget, 0) + 50 WHERE owner_username=%s",
                                (stripe_sess.client_reference_id,),
                            )
            except Exception:
                pass
        return RedirectResponse("/dashboard?ppc_success=true", status_code=302)

    @app.get("/payment-success")
    def payment_success(session_id: str):
        sk = get_stripe_secret()
        if sk:
            stripe.api_key = sk
            try:
                stripe_sess = stripe.checkout.Session.retrieve(session_id)
                if stripe_sess.payment_status == "paid":
                    expiry = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
                    with get_db() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE companies SET is_premium=TRUE, premium_expiry=%s WHERE owner_username=%s",
                                (expiry, stripe_sess.client_reference_id),
                            )
            except Exception:
                pass
        return RedirectResponse("/dashboard?success=true", status_code=302)

    @api.get("/config")
    def api_config():
        return {
            "stripe_pk": get_stripe_pub(),
            "maps_key": os.environ.get("GOOGLE_MAPS_API_KEY", ""),
        }

    @api.get("/health")
    def health():
        gemini_key = os.environ.get("GEMINI_GOOGLE_API_KEY") or os.environ.get("GOOGLE_GEMINI_API_KEY")
        maps_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
        db_ok = False
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            db_ok = True
        except Exception:
            pass
        return {
            "ok": db_ok,
            "db": db_ok,
            "stripe": bool(get_stripe_secret()),
            "ai_gemini": bool(gemini_key),
            "maps": bool(maps_key),
        }

    app.include_router(api, prefix="/api")
    app.mount("/static", StaticFiles(directory=static_dir), name="ui")
    return app
