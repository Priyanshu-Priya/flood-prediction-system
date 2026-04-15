"""
Copernicus CDSE OAuth2 Authentication Manager
=============================================
Handles token acquisition, caching, and automatic refreshing for 
CDSE (Copernicus Data Space Ecosystem) services.

Follows the OAuth2 Client Credentials flow.
"""

import os
import time
import requests
from dotenv import load_dotenv
from loguru import logger
from typing import Optional, Dict

from config.settings import settings

# Load at startup as per best practice
load_dotenv()

class CopernicusTokenManager:
    """
    Manages OAuth2 access tokens for Copernicus Data Space Ecosystem.
    Implements in-memory caching to avoid redundant token generation.
    """
    
    _instance = None
    _token_cache: Dict[str, any] = {
        "token": None,
        "expiry": 0
    }

    def __new__(cls):
        """Singleton pattern for shared cache."""
        if cls._instance is None:
            cls._instance = super(CopernicusTokenManager, cls).__new__(cls)
        return cls._instance

    def validate_env(self):
        """
        Agent Validation Layer: Ensures critical secrets are present in environment.
        Strictly strictly enforces the 'Never hardcode secrets' rule.
        """
        required = ["COPERNICUS_CLIENT_ID", "COPERNICUS_CLIENT_SECRET"]
        for var in required:
            val = os.getenv(var)
            if not val:
                raise ValueError(f"CRITICAL: {var} not found in environment variables. Access denied.")
        
        # Masked logging to verify load without exposing secret
        client_id = os.getenv("COPERNICUS_CLIENT_ID")
        logger.info(f"Copernicus Environment Validated | Client ID: {client_id[:10]}...")

    def get_access_token(self, force_refresh: bool = False) -> str:
        """
        Get a valid access token. Returns cached token if still valid,
        otherwise requests a new one from the CDSE identity provider.
        """
        now = time.time()
        
        # Use cached token if it's still valid (with 60-second buffer)
        if not force_refresh and self._token_cache["token"] and now < (self._token_cache["expiry"] - 60):
            logger.debug("Using cached Copernicus access token")
            return self._token_cache["token"]

        return self._refresh_token()

    def _refresh_token(self) -> str:
        """Fetch a new token from the Copernicus Identity server."""
        logger.info("Refreshing Copernicus access token...")
        
        url = settings.data_sources.copernicus_auth_url
        
        # Always fetch from environment using os.getenv()
        # Never pass variable names as strings in requests
        client_id = os.getenv("COPERNICUS_CLIENT_ID")
        client_secret = os.getenv("COPERNICUS_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise Exception("Missing Copernicus credentials in environment variables. Authentication aborted.")

        if not client_id or not client_secret:
            raise ValueError(
                "Copernicus credentials missing! Set COPERNICUS_CLIENT_ID "
                "and COPERNICUS_CLIENT_SECRET in .env"
            )

        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        try:
            response = requests.post(url, data=data, headers=headers, timeout=15)
            response.raise_for_status()
            
            payload = response.json()
            token = payload["access_token"]
            expires_in = payload.get("expires_in", 1800)  # Default 30 min if missing
            
            # Update cache
            self._token_cache["token"] = token
            self._token_cache["expiry"] = time.time() + expires_in
            
            logger.info(f"Copernicus token refreshed successfully (expires in {expires_in}s)")
            return token

        except Exception as e:
            logger.error(f"Failed to fetch Copernicus access token: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise RuntimeError(f"Authentication with Copernicus CDSE failed: {e}")

    def verify_glofas_access(self, execution_url: str) -> bool:
        """
        Verify if the GloFAS dataset is activated for the current account.
        Sends a minimal, valid payload to the execution endpoint.
        """
        token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Minimal payload for permission check (1 station, 1 day)
        payload = {
            "system_version": "version_4_0",
            "hydrological_model": "lisflood",
            "product_type": "consolidated",
            "variable": "river_discharge_in_the_last_24_hours",
            "hyear": "2020",
            "hmonth": "01",
            "hday": ["01"],
            "area": [26.0, 91.0, 25.0, 92.0], # Small bbox
            "download_format": "zip"
        }

        try:
            logger.info("Verifying GloFAS dataset activation status...")
            response = requests.post(execution_url, headers=headers, json=payload, timeout=20)
            
            error_code = self.handle_api_error(response)
            if error_code == "DATASET_NOT_ACTIVATED":
                logger.warning("GloFAS dataset is NOT activated for this account.")
                return False
            
            if response.status_code == 201 or response.status_code == 200:
                logger.info("GloFAS dataset access verified (Activated)")
                return True
                
            logger.error(f"Unexpected response during GloFAS verification: {response.status_code}")
            return False

        except Exception as e:
            logger.error(f"Error during GloFAS activation check: {e}")
            return False

    @staticmethod
    def handle_api_error(response: requests.Response) -> Optional[str]:
        """
        Classify Copernicus API errors into known categories.
        """
        if response.status_code in [200, 201]:
            return None
            
        try:
            text = response.text.lower()
            if "operation not allowed" in text:
                return "DATASET_NOT_ACTIVATED"
            if "request too large" in text or "cost limit" in text:
                return "REQUEST_TOO_LARGE"
            if response.status_code == 401:
                return "AUTH_FAILURE"
        except:
            pass
            
        return "UNKNOWN_ERROR"

# Global singleton for easy access
copernicus_auth = CopernicusTokenManager()
