"""
google_auth.py — Shared Google credential loader (Service Account).

All Google API clients (Drive, Firestore) use the same service account key file.
No OAuth dance, no token expiry, no scope drift.

Key file path: configured via [google] service_account_path in teledigest.conf.
Default:       /home/teledigest/data/service-account.json
"""

from __future__ import annotations

from pathlib import Path

from .config import get_config, log


def load_service_account_credentials(scopes: list[str]):
    """
    Load Google service account credentials from the configured key file.

    Raises FileNotFoundError if the key file is missing.
    """
    from google.oauth2 import service_account

    cfg = get_config()
    sa_path = cfg.google.service_account_path

    if not sa_path.exists():
        raise FileNotFoundError(
            f"Service account key not found: {sa_path}. "
            "Download a JSON key from GCP Console → IAM & Admin → Service Accounts "
            "and place it at that path (or set [google] service_account_path in config)."
        )

    return service_account.Credentials.from_service_account_file(
        str(sa_path), scopes=scopes
    )


def build_firestore_client():
    """Build an authenticated Firestore client using service account."""
    from google.cloud import firestore

    cfg = get_config()
    if not cfg.google.firestore_project_id:
        raise RuntimeError("[google] firestore_project_id is not set in config.")

    creds = load_service_account_credentials(
        ["https://www.googleapis.com/auth/datastore"]
    )
    return firestore.Client(
        project=cfg.google.firestore_project_id,
        database=cfg.google.firestore_database,
        credentials=creds,
    )
