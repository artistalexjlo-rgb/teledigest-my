#!/usr/bin/env python3
"""
drive_oauth_init.py — One-shot OAuth flow to mint google-token.json.

Run this **on your local machine** (not on the VPS — needs a browser).

Usage:
    python scripts/drive_oauth_init.py path/to/google-credentials.json

Then copy both google-credentials.json and the produced google-token.json
to the VPS (Coolify mount alongside the SQLite DB), and set their paths in
teledigest.conf [google] section.

Why a script instead of running this in the bot:
- The OAuth flow opens a browser to ask consent. The bot runs headless on a
  VPS, so we do this once locally; the resulting refresh_token in token.json
  lets the bot operate autonomously afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: drive_oauth_init.py <path-to-credentials.json> [output-token.json]",
            file=sys.stderr,
        )
        return 2

    creds_path = Path(sys.argv[1]).expanduser()
    if not creds_path.is_file():
        print(f"credentials file not found: {creds_path}", file=sys.stderr)
        return 1

    token_path = (
        Path(sys.argv[2]).expanduser() if len(sys.argv) > 2
        else creds_path.with_name("google-token.json")
    )

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(
            "Missing dependency. Run:\n"
            "  pip install google-auth-oauthlib google-api-python-client",
            file=sys.stderr,
        )
        return 1

    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
    # port=0 picks a free port automatically; opens default browser.
    creds = flow.run_local_server(port=0)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    print(f"OK: token saved to {token_path}")
    print(
        "Next: copy both files to the VPS (Coolify mount, e.g. "
        "/home/teledigest/data/) and update teledigest.conf [google]."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
