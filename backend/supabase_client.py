import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def get_supabase_url() -> str:
    value = os.getenv("SUPABASE_URL", "").strip()
    if not value:
        raise RuntimeError("SUPABASE_URL is not configured. Set it in backend .env.")
    return value


def get_supabase_service_role_key() -> str:
    value = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not value:
        value = os.getenv("SUPABASE_SECRET_KEY", "").strip()
    if not value:
        raise RuntimeError(
            "SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SECRET_KEY is not configured. "
            "Set one in the backend .env."
        )
    return value


def get_supabase_client() -> Any:
    from supabase import create_client

    url = get_supabase_url()
    key = get_supabase_service_role_key()
    return create_client(url, key)


def get_table_rows(table_name: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """Read a bounded set of rows using the backend-only Supabase client."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    response = get_supabase_client().table(table_name).select("*").limit(limit).execute()
    return response.data or []


# Import this name in backend modules after configuring Supabase credentials.
supabase = get_supabase_client()
