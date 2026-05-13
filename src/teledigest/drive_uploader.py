"""
drive_uploader.py — Push local files to a Google Drive folder.

Auth: Service Account (JSON key file). No OAuth dance, no token expiry.
Key file path: [google] service_account_path in teledigest.conf.
Default:       /home/teledigest/data/service-account.json

The Drive folder must be shared with the service account email (Editor role).

Idempotency: when uploading a file whose name already exists in the target
folder, the existing file's content is replaced (revision update). This makes
re-runs safe — the same daily sample file overwrites itself instead of
accumulating duplicates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .config import get_config, log

DRIVE_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
)


def get_drive_service():
    """Build an authenticated Drive v3 service using service account."""
    from googleapiclient.discovery import build
    from .google_auth import load_service_account_credentials

    cfg = get_config()
    if not cfg.google.enabled:
        raise RuntimeError("Drive upload is disabled (no [google] config).")

    creds = load_service_account_credentials(list(DRIVE_SCOPES))
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Upload primitives
# ---------------------------------------------------------------------------

def _find_existing_file_id(service, name: str, folder_id: str) -> str | None:
    """
    Look up a file by exact name within a parent folder. Returns id or None.

    Drive query syntax: name match + parent constraint + non-trashed.
    Names with apostrophes need escaping; daily-sample names never have them
    (date + country + slug only), so we keep it simple.
    """
    q = f"name = '{name}' and '{folder_id}' in parents and trashed = false"
    res = service.files().list(
        q=q, fields="files(id, name)", pageSize=10, spaces="drive",
    ).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None


def upload_file(service, local_path: Path, folder_id: str) -> tuple[str, bool]:
    """
    Upload `local_path` into Drive folder `folder_id`.

    If a file with the same name already lives in that folder, its content
    is replaced (revision update). Otherwise a new file is created.

    Returns (file_id, created) — `created=False` means revision update.
    """
    from googleapiclient.http import MediaFileUpload

    name = local_path.name
    media = MediaFileUpload(
        str(local_path), mimetype="text/plain", resumable=False,
    )
    existing_id = _find_existing_file_id(service, name, folder_id)
    if existing_id:
        f = service.files().update(
            fileId=existing_id, media_body=media, fields="id",
        ).execute()
        return f["id"], False

    body = {"name": name, "parents": [folder_id]}
    f = service.files().create(
        body=body, media_body=media, fields="id",
    ).execute()
    return f["id"], True


def upload_files(
    service, paths: Iterable[Path], folder_id: str,
) -> list[tuple[Path, str | None, bool]]:
    """
    Upload multiple files. Returns list of (path, drive_id_or_None, created).

    A None drive_id signals that the upload failed for that file (logged).
    """
    results: list[tuple[Path, str | None, bool]] = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            fid, created = upload_file(service, p, folder_id)
            log.info(
                "Drive upload: %s -> id=%s %s",
                p.name, fid, "(created)" if created else "(updated)",
            )
            results.append((p, fid, created))
        except Exception as e:
            log.error("Drive upload failed for %s: %s", p, e)
            results.append((p, None, False))
    return results


# ---------------------------------------------------------------------------
# Public entry: upload a directory of sample files
# ---------------------------------------------------------------------------

def upload_samples_dir(samples_dir: Path | None = None) -> list[tuple[Path, str | None, bool]]:
    """
    Walk the daily-samples directory and push every .txt file to Drive.

    Returns the per-file result list. Safe to call when Drive isn't configured —
    logs a warning and returns []. Never raises to caller (so a Drive outage
    can't kill the scheduler).
    """
    cfg = get_config()
    if not cfg.google.enabled:
        log.info("Drive upload skipped — [google] not enabled in config.")
        return []

    if samples_dir is None:
        from .daily_samples import get_samples_dir
        samples_dir = get_samples_dir()

    if not samples_dir.exists():
        log.warning("Drive upload: samples dir %s does not exist.", samples_dir)
        return []

    files = sorted(samples_dir.rglob("*.txt"))
    if not files:
        log.info("Drive upload: no .txt files in %s — nothing to send.", samples_dir)
        return []

    try:
        service = get_drive_service()
    except Exception as e:
        log.error("Drive upload aborted — auth failed: %s", e)
        return []

    return upload_files(service, files, cfg.google.drive_folder_id)
