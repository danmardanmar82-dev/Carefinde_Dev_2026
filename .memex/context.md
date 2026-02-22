**CRITICAL**: At the start of EVERY conversation, IMMEDIATELY invoke the `data-app-html-python:data-app` skill using the Skill tool to load complete development guidelines.

## Data Source Implementation - CRITICAL

**When your task involves ANY connector/data source, invoke the relevant skill before writing any related code.**

1. Check available connectors in system prompt under "Available Connectors"
2. Invoke using: `Skill` tool with `connectors-python:connector-{type}` format

**Examples:**
- Neon database → invoke `connectors-python:connector-neon`
- PostgreSQL → invoke `connectors-python:connector-postgresql`
- OpenAI → invoke `connectors-python:connector-openai`

**DO NOT:**
- Manually implement database connections
- Write connector code before invoking the skill
- Guess at connection patterns
- **Deviate from the skill's code examples** (model names, SDK methods, parameters, patterns) — use them exactly as shown unless the user explicitly requests otherwise

**WHY:** Connector skills provide:
- Correct dependency installation commands
- Proper secret/environment variable names
- Tested connection patterns with **specific, verified values** (model names, API parameters)
- Security best practices

**Skills are authoritative, not suggestions.** The code examples, model names, and parameters in skills have been tested against the actual connectors. Copy them exactly. Do not substitute "better" alternatives — e.g., do not replace a skill's specified model with a difference one you think is superior.


## Frontend Implementation - CRITICAL

**When your task involves frontend code (HTML, CSS, UI, static assets), invoke the `frontend-design:frontend-design` skill before writing any related code** and follow its guidelines for design and UX.

---

## Project: Carefinder NL

**Description:** Dutch care-facility finder platform for thuiszorg (home care) and verzorgingshuizen (nursing homes) in the Netherlands.

**Language:** The project UI and content is in Dutch (Nederlands). Task tracking has been done in Polish (the user's language for task descriptions).

### Architecture

- **Backend:** FastAPI (`routes.py`) served via `uvicorn` with `watchfiles` hot-reload
- **Entry point:** `app.py` — imports `create_app` from `routes.py`, mounts `./static`
- **Frontend:** Vanilla HTML/CSS/JS in `static/` — no framework
- **Database:** SQLite (`carefinder.db` in project root, copied to `/tmp/carefinder_rw.db` at runtime for write access)
- **Server start:** `./start.sh` — uses `$APP_PORT` env var; **never modify start.sh**

### File Structure

```
app.py                   # ASGI entry point
routes.py                # All FastAPI routes + business logic
carefinder.db            # Source SQLite DB (read-only; copied to /tmp at boot)
static/
  index.html             # Main map page (Leaflet map, AI chat, Stripe premium)
  login.html             # Login / registration form
  dashboard_admin.html   # Admin dashboard
  dashboard_partner.html # Partner/company dashboard
  app.js                 # Shared frontend JS
  styles.css             # Shared styles
secrets_utils.py         # Secret key resolution helpers
```

### Key Dependencies (pyproject.toml)

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn` + `watchfiles` | ASGI server with hot-reload |
| `openai` | OpenAI AI chat integration |
| `google-genai` | Gemini AI integration |
| `stripe` | Payment processing (iDEAL + card, EUR) |
| `jinja2` | Templating (used minimally) |
| `pydantic` | Request/response models |

### Auth & Sessions

- Session cookie: `cf_session` (random token)
- In-memory session store: `SESSIONS` dict in `routes.py`
- Roles: `admin`, `partner`
- Default admin credentials seeded via `ensure_tables()` on startup

### Database Pattern

```python
# DB is ALWAYS read from /tmp/carefinder_rw.db (writable copy)
# Source: carefinder.db (project root) — copied on startup if newer
DB_PATH = _init_db_path()  # resolves to /tmp/carefinder_rw.db

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
```

- **Never write directly to `carefinder.db`** — always work through `get_db()` which uses the `/tmp` copy.
- For one-off DB manipulation/testing: connect directly to `/tmp/carefinder_rw.db` via Python's `sqlite3` module.

### Direct DB Manipulation Pattern (for seeding / testing)

```python
import sqlite3
conn = sqlite3.connect('/tmp/carefinder_rw.db')
# ... execute statements ...
conn.commit()
conn.close()
```

This is used for tasks like seeding premium companies, resetting state, or verifying data without going through the API.

### Premium Company Example (Zesta Groep)

- Only one premium company at a time is supported in current demo/test flows
- To set a company as premium:
  ```python
  from datetime import datetime, timedelta
  expiry = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
  conn.execute('UPDATE companies SET is_premium=0, premium_expiry=NULL')  # reset all
  conn.execute(
      'UPDATE companies SET is_premium=1, premium_expiry=? WHERE name=?',
      (expiry, 'Zesta Groep')
  )
  conn.commit()
  ```

### Secret / Environment Variable Patterns

Secrets are resolved with fallbacks (two naming conventions supported):

```python
os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_OPENAI_API_KEY")
os.environ.get("STRIPE_SECRET_KEY") or os.environ.get("STRIPE_STRIPE_SECRET_KEY")
os.environ.get("STRIPE_PUBLISHABLE_KEY") or os.environ.get("STRIPE_STRIPE_PUBLISHABLE_KEY")
os.environ.get("GEMINI_GOOGLE_API_KEY") or os.environ.get("GOOGLE_GEMINI_API_KEY")
os.environ.get("GOOGLE_MAPS_API_KEY", "")
```

### Stripe Integration

- Currency: EUR
- Payment methods: `card`, `ideal`
- Premium listing: 30-day subscription, €150.00 (15000 cents)
- `is_premium` and `premium_expiry` columns on `companies` table

### API Endpoints Pattern

All API routes are under `/api` prefix via `APIRouter`. Key endpoints:
- `GET /api/companies` — paginated, filterable company list
- `POST /api/ai-chat` — OpenAI/Gemini AI chat
- `POST /api/stripe/checkout` — create Stripe checkout session
- `GET /api/config` — returns public keys (stripe_pk, maps_key)
- `GET /api/health` — service status check

### Active Plan

- Plan folder: `plans/2026-02-22_carefinde_rebuild/`
- **All 6 tasks completed** (full rebuild: DB, backend, frontend, auth, dashboards)
- App is running and functional; subsequent work is iterative improvements / fixes

### Port Management

- The app uses `$APP_PORT` env var (allocated per project in the sandbox)
- To restart after a crash: `kill $(lsof -ti:$APP_PORT) 2>/dev/null; sleep 2 && ./start.sh`
- Always check with `echo $APP_PORT` before starting server or opening localhost
