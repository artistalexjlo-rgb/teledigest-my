"""
create_vector_indexes.py — Create Firestore vector indexes for embeddings.

Firebase CLI does not support vectorConfig in firestore.indexes.json yet.
This script creates them via the Firestore Admin REST API using the same
service account credentials as the bot.

Usage:
    python scripts/create_vector_indexes.py

Requires GOOGLE_APPLICATION_CREDENTIALS or google-token.json in the repo root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

PROJECT_ID = "project-56cb62a9-8914-4ae3-b44"
DATABASE = "(default)"

VECTOR_INDEXES = [
    {"collection": "wisdom_base", "field": "embedding", "dimension": 768},
    {"collection": "wikivoyage_base", "field": "embedding", "dimension": 768},
]


def _get_access_token() -> str:
    """Get OAuth2 access token from google-token.json (same as bot uses)."""
    token_paths = [
        Path(__file__).parent.parent / "google-token.json",
        Path(__file__).parent.parent / "token.json",
    ]
    for p in token_paths:
        if p.exists():
            data = json.loads(p.read_text())
            token = data.get("access_token") or data.get("token")
            if token:
                print(f"Using token from {p}")
                return token

    # Try google.auth default credentials (service account via env)
    try:
        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(google.auth.transport.requests.Request())
        return creds.token
    except Exception as e:
        print(f"google.auth failed: {e}")

    print("ERROR: No credentials found.")
    print("Run: python -m teledigest.scripts.auth  OR  set GOOGLE_APPLICATION_CREDENTIALS")
    sys.exit(1)


def create_vector_index(token: str, collection: str, field: str, dimension: int) -> None:
    url = (
        f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}"
        f"/databases/{DATABASE}/collectionGroups/{collection}/indexes"
    )
    body = {
        "queryScope": "COLLECTION",
        "fields": [
            {
                "fieldPath": field,
                "vectorConfig": {
                    "dimension": dimension,
                    "flat": {},
                },
            }
        ],
    }
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=30,
    )
    if resp.status_code in (200, 201):
        op = resp.json().get("name", "")
        print(f"  OK — operation: {op}")
        print(f"  Index is building async, check: https://console.firebase.google.com/project/{PROJECT_ID}/firestore/indexes")
    elif resp.status_code == 409:
        print(f"  ALREADY EXISTS — skipped")
    else:
        print(f"  ERROR {resp.status_code}: {resp.text}")


def main() -> None:
    print("Creating Firestore vector indexes...")
    token = _get_access_token()
    for idx in VECTOR_INDEXES:
        print(f"\n{idx['collection']}.{idx['field']} ({idx['dimension']}d):")
        create_vector_index(token, idx["collection"], idx["field"], idx["dimension"])
    print("\nDone. Indexes build in the background (usually 1-5 min).")


if __name__ == "__main__":
    main()
