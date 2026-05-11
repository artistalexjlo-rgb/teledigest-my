#!/usr/bin/env python3
"""
drive_oauth_init.py — One-shot OAuth flow to mint google-token.json.

Run this **on your local machine** (not on the VPS — needs a browser).

The token grants access to THREE Google APIs that the bot uses with one
shared OAuth identity:
  - Drive (upload daily samples, list files in the target folder)
  - Firestore (read wisdom_base + telegram_queue, write wikivoyage_base)
  - (datastore scope is the Firestore programmatic name)

If a feature stops working with `403 Insufficient Permission` or
`insufficientPermissions`, the token is missing one of these scopes —
re-run this script and replace the file on the VPS.

Usage:
    python scripts/drive_oauth_init.py path/to/google-credentials.json

    # if local-server flow fails (firewall, port conflict):
    python scripts/drive_oauth_init.py path/to/credentials.json --console

Then copy the produced google-token.json to the VPS (e.g. Dokploy file
mount alongside the SQLite DB, /home/teledigest/data/google-token.json)
and restart the container.

Why a script instead of running this in the bot:
- The OAuth flow opens a browser to ask consent. The bot runs headless on a
  VPS, so we do this once locally; the resulting refresh_token in token.json
  lets the bot operate autonomously afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Scopes mint EVERYTHING this bot's google-token.json is used for:
#   drive.file  — drive_uploader uploads/updates daily-sample TXT files
#   drive       — drive_uploader.files.list() to dedupe uploads. drive.file
#                 alone is not enough for searching by name in a folder that
#                 was created outside this app's OAuth context.
#   datastore   — Firestore access (channel_poster, gemini_brain МОЗГ,
#                 wikivoyage_import). All three reuse the SAME token.
#
# If you change this list, EVERY existing google-token.json must be
# re-minted by running this script again and copying the new file to VPS.
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/datastore",
]


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

    use_console = "--console" in sys.argv

    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
    if use_console:
        # Manual flow: print URL, user opens in any browser, pastes code back.
        # No local server, no redirect — bullet-proof against firewall/port issues.
        flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
        print("\nOpen this URL in any browser, sign in, click Allow:\n")
        print(auth_url)
        print("\nAfter Allow you'll see a code on the page. Paste it below.")
        code = input("\nAuthorization code: ").strip()
        flow.fetch_token(code=code)
        creds = flow.credentials
    else:
        # Local-server flow (default): opens browser automatically and listens
        # on a random port for the OAuth redirect. If your firewall blocks it
        # or the browser hangs, re-run with --console flag for manual paste.
        try:
            creds = flow.run_local_server(port=0, timeout_seconds=180, open_browser=True)
        except Exception as e:
            print(
                f"\nLocal server flow failed ({e}).\n"
                "Re-run with --console flag for manual copy-paste flow.",
                file=sys.stderr,
            )
            return 1
    token_path.write_text(creds.to_json(), encoding="utf-8")
    print(f"OK: token saved to {token_path}")
    print(
        "Next: copy both files to the VPS (Coolify mount, e.g. "
        "/home/teledigest/data/) and update teledigest.conf [google]."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
