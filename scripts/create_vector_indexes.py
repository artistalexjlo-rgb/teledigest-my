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

import sys
from pathlib import Path

import requests

PROJECT_ID = "project-56cb62a9-8914-4ae3-b44"
DATABASE = "default"

VECTOR_INDEXES = [
    # Cipher-fix uses gemini-embedding-2 truncated to 1536 (Firestore vector
    # index caps at 2048; 1536 is the largest MRL-supported point).
    # Old 768-dim indexes on these same collections must be DROPPED first via
    # console / API — Firestore won't accept a second vector index on the
    # same field-path. See roadmap_embedding_cipher_fix.md Phase 1.6.
    {"collection": "wisdom_base", "field": "embedding", "dimension": 1536},
    {"collection": "wikivoyage_base", "field": "embedding", "dimension": 1536},
]


def _get_access_token() -> str:
    """
    Get OAuth2 access token with cloud-platform scope via installed-app flow.
    Opens browser for one-time consent, then returns the token.
    Does NOT overwrite google-token.json — uses a temp file.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

    creds_paths = [
        Path(__file__).parent.parent / "credentials.json",
        Path(__file__).parent.parent / "google-credentials.json",
        Path(__file__).parent.parent / "scripts" / "credentials.json",
    ]
    creds_file = next((p for p in creds_paths if p.exists()), None)
    if not creds_file:
        print("ERROR: credentials.json not found")
        sys.exit(1)

    print(f"Using credentials from {creds_file}")
    print("A browser window will open for Google auth (one-time)...")
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)
    creds = flow.run_local_server(port=0)
    return creds.token


def create_vector_index(
    token: str, collection: str, field: str, dimension: int
) -> None:
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
        print(
            f"  Index is building async, check: https://console.firebase.google.com/project/{PROJECT_ID}/firestore/indexes"
        )
    elif resp.status_code == 409:
        print("  ALREADY EXISTS — skipped")
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
