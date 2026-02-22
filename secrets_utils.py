"""Utility for OAuth access token retrieval."""
import os
import httpx

def get_oauth_access_token(connector_id: str) -> str:
    """Fetch fresh OAuth access token from Modal backend.

    This function is used by the apps to get access tokens for
    OAuth connectors like Google Drive. The access token is short-lived
    (~1 hour) and should be fetched fresh when needed.

    Args:
        connector_id: The OAuth connector ID (e.g., from os.environ.get("CONNECTORPREFIX_CONNECTOR_ID")                     

    Returns:
        Access token string for use with provider APIs (Google Drive, etc.)

    Raises:
        ValueError: If required environment variables are not set
        httpx.HTTPStatusError: If the backend request fails

    Example:
        >>> connector_id = os.environ.get("CONNECTORPREFIX_CONNECTOR_ID")   # UUID
        >>> access_token = get_oauth_access_token(connector_id)
        >>> # Use with Google Drive API
        >>> from google.oauth2.credentials import Credentials
        >>> creds = Credentials(token=access_token)

    Note:
        MEMEX_DEPLOYMENT_TOKEN and MEMEX_BACKEND_URL are auto-injected
        by the deployment system when OAuth connectors are included in
        the deployment. No manual configuration is needed.
    """
    token = os.environ.get("MEMEX_DEPLOYMENT_TOKEN")
    backend_url = os.environ.get("MEMEX_BACKEND_URL")

    if not token:
        raise ValueError(
            "MEMEX_DEPLOYMENT_TOKEN not set. "
            "Ensure OAuth connectors are included in the deployment."
        )
    if not backend_url:
        raise ValueError(
            "MEMEX_BACKEND_URL not set. "
            "Ensure OAuth connectors are included in the deployment."
        )

    response = httpx.get(
        f"{backend_url}/deployments/connectors/{connector_id}/access_token",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )
    response.raise_for_status()

    data = response.json()
    return data["access_token"]