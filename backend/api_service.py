import datetime
import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, ".env"))


class ExternalAPIError(RuntimeError):
    """A safe, user-facing external API failure without credential details."""


class ExternalAPIService:
    def __init__(self) -> None:
        self.url = os.getenv("EXTERNAL_API_URL", "").strip()
        self.api_key = os.getenv("EXTERNAL_API_KEY", "").strip()
        self.auth_header = os.getenv("EXTERNAL_API_AUTH_HEADER", "Authorization").strip()
        self.auth_scheme = os.getenv("EXTERNAL_API_AUTH_SCHEME", "Bearer").strip()
        self.timeout_seconds = float(os.getenv("EXTERNAL_API_TIMEOUT_SECONDS", "10"))

    @property
    def configured(self) -> bool:
        return bool(self.url and self.api_key)

    def fetch_location_data(self, *, latitude: float, longitude: float) -> Dict[str, Any]:
        if not self.configured:
            return {"status": "unavailable", "reason": "External API is not configured."}

        headers = {self.auth_header: f"{self.auth_scheme} {self.api_key}" if self.auth_scheme else self.api_key}
        try:
            response = requests.get(
                self.url,
                params={"latitude": latitude, "longitude": longitude},
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise ExternalAPIError("External API request timed out.") from exc
        except requests.RequestException as exc:
            raise ExternalAPIError("External API request failed.") from exc
        except ValueError as exc:
            raise ExternalAPIError("External API returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise ExternalAPIError("External API returned an invalid response shape.")

        return {
            "status": "available",
            "fetched_at": datetime.datetime.utcnow().isoformat() + "Z",
            "data": payload,
        }


external_api_service = ExternalAPIService()
