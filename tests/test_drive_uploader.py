"""Tests for drive_uploader. Mocks the Drive service entirely — no network."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from teledigest import config as cfg
from teledigest import drive_uploader as du


def _make_app_config(
    db_path: Path,
    *,
    drive_enabled: bool = True,
    folder_id: str = "FOLDER123",
    token_path: Path | None = None,
    creds_path: Path | None = None,
) -> cfg.AppConfig:
    return cfg.AppConfig(
        telegram=cfg.TelegramConfig(
            api_id=1, api_hash="h", bot_token="t", sessions_dir=Path("td")
        ),
        bot=cfg.BotConfig(channels=["@c"], summary_target="@d"),
        llm=cfg.LLMConfig(model="x", api_key="k", system_prompt="", user_prompt=""),
        storage=cfg.StorageConfig(rag_keywords=[], db_path=db_path),
        logging=cfg.LoggingConfig(level="INFO"),
        google=cfg.GoogleConfig(
            drive_folder_id=folder_id if drive_enabled else "",
            credentials_path=creds_path or Path("creds.json"),
            token_path=token_path or Path("token.json"),
            enabled=drive_enabled,
        ),
    )


@pytest.fixture
def app_config(tmp_path, monkeypatch) -> cfg.AppConfig:
    app_cfg = _make_app_config(
        db_path=tmp_path / "messages_fts.db",
        token_path=tmp_path / "token.json",
        creds_path=tmp_path / "creds.json",
    )
    monkeypatch.setattr(cfg, "_CONFIG", app_cfg, raising=False)
    return app_cfg


# ---------------------------------------------------------------------------
# Upload primitives
# ---------------------------------------------------------------------------

def _make_fake_service(existing_files: dict[tuple[str, str], str] | None = None):
    """
    Build a MagicMock that mimics the subset of Drive API we use.

    `existing_files` maps (name, folder_id) -> existing file id (so list()
    returns it for that pair). create()/update() always succeed.
    """
    existing = existing_files or {}
    files_api = MagicMock()

    def list_side_effect(q, fields, pageSize, spaces):
        # Parse our query format: "name = 'X' and 'F' in parents and trashed = false"
        # Quick brittle parser is fine for tests.
        import re
        m = re.search(r"name = '([^']+)' and '([^']+)' in parents", q)
        if not m:
            return MagicMock(execute=lambda: {"files": []})
        name, folder = m.group(1), m.group(2)
        fid = existing.get((name, folder))
        result = {"files": [{"id": fid, "name": name}] if fid else []}
        return MagicMock(execute=lambda: result)

    files_api.list = MagicMock(side_effect=list_side_effect)

    create_call = MagicMock()
    create_call.execute = MagicMock(return_value={"id": "NEW_ID_123"})
    files_api.create = MagicMock(return_value=create_call)

    update_call = MagicMock()
    update_call.execute = MagicMock(return_value={"id": "UPDATED_ID"})
    files_api.update = MagicMock(return_value=update_call)

    service = MagicMock()
    service.files = MagicMock(return_value=files_api)
    return service, files_api


def test_upload_file_creates_when_not_exists(tmp_path, app_config):
    sample = tmp_path / "2026-04-30_br_Brazil_ChatForum.txt"
    sample.write_text("hello", encoding="utf-8")
    service, files_api = _make_fake_service(existing_files={})

    with patch.object(du, "MediaFileUpload" if hasattr(du, "MediaFileUpload") else "_NA",
                      MagicMock(), create=True):
        # Patch the import inside upload_file
        with patch("googleapiclient.http.MediaFileUpload") as mock_media:
            fid, created = du.upload_file(service, sample, "FOLDER123")

    assert fid == "NEW_ID_123"
    assert created is True
    files_api.create.assert_called_once()
    files_api.update.assert_not_called()
    # Body included name and parent folder
    create_kwargs = files_api.create.call_args.kwargs
    assert create_kwargs["body"]["name"] == "2026-04-30_br_Brazil_ChatForum.txt"
    assert create_kwargs["body"]["parents"] == ["FOLDER123"]


def test_upload_file_updates_when_exists(tmp_path, app_config):
    sample = tmp_path / "2026-04-30_br_Brazil_ChatForum.txt"
    sample.write_text("hello v2", encoding="utf-8")
    service, files_api = _make_fake_service(existing_files={
        ("2026-04-30_br_Brazil_ChatForum.txt", "FOLDER123"): "EXISTING_ID_42",
    })

    with patch("googleapiclient.http.MediaFileUpload"):
        fid, created = du.upload_file(service, sample, "FOLDER123")

    assert fid == "UPDATED_ID"
    assert created is False
    files_api.update.assert_called_once()
    files_api.create.assert_not_called()
    update_kwargs = files_api.update.call_args.kwargs
    assert update_kwargs["fileId"] == "EXISTING_ID_42"


def test_upload_files_continues_on_per_file_failure(tmp_path, app_config):
    """
    Mini-pipeline scenario:
      Setup: 3 files, middle one will trigger Drive API error.
      Action: upload_files
      Expectation: first and last succeed; middle returns (path, None, False);
                   loop didn't crash.
    """
    a = tmp_path / "a.txt"; a.write_text("a")
    b = tmp_path / "b.txt"; b.write_text("b")
    c = tmp_path / "c.txt"; c.write_text("c")
    service, files_api = _make_fake_service()

    # Make the 2nd create() raise
    create_call_ok = MagicMock(execute=MagicMock(return_value={"id": "ID_OK"}))
    create_call_fail = MagicMock(execute=MagicMock(side_effect=RuntimeError("boom")))
    files_api.create = MagicMock(side_effect=[create_call_ok, create_call_fail, create_call_ok])

    with patch("googleapiclient.http.MediaFileUpload"):
        results = du.upload_files(service, [a, b, c], "FOLDER123")

    assert len(results) == 3
    assert results[0][1] == "ID_OK"
    assert results[0][2] is True
    assert results[1][1] is None  # failed file — no id, but loop continued
    assert results[2][1] == "ID_OK"


# ---------------------------------------------------------------------------
# upload_samples_dir end-to-end (the entry point scheduler calls)
# ---------------------------------------------------------------------------

def test_upload_samples_dir_skips_when_disabled(app_config, tmp_path, monkeypatch):
    # Re-make config with disabled Drive
    cfg_disabled = _make_app_config(
        db_path=tmp_path / "messages_fts.db", drive_enabled=False,
    )
    monkeypatch.setattr(cfg, "_CONFIG", cfg_disabled, raising=False)
    samples = tmp_path / "samples" / "br"
    samples.mkdir(parents=True)
    (samples / "2026-04-30_br_X.txt").write_text("data")

    results = du.upload_samples_dir(samples_dir=tmp_path / "samples")
    assert results == []


def test_upload_samples_dir_uploads_every_txt(app_config, tmp_path, monkeypatch):
    """
    Mini-pipeline scenario:
      Setup: 3 sample files in samples/{country}/.
      Action: upload_samples_dir() with mocked Drive service.
      Expectation: each file produced an upload_file call with the right
                   folder_id; non-.txt files in dir are ignored.
    """
    samples_root = tmp_path / "samples"
    (samples_root / "br").mkdir(parents=True)
    (samples_root / "id").mkdir(parents=True)
    (samples_root / "br" / "2026-04-30_br_Brazil_ChatForum.txt").write_text("br1")
    (samples_root / "br" / "2026-04-30_br_1001631614451.txt").write_text("br2")
    (samples_root / "id" / "2026-04-30_id_balichat.txt").write_text("id1")
    # Junk: not a .txt — must be ignored
    (samples_root / "br" / "ignore_me.json").write_text("{}")

    fake_service, files_api = _make_fake_service()

    with patch.object(du, "get_drive_service", return_value=fake_service), \
         patch("googleapiclient.http.MediaFileUpload"):
        results = du.upload_samples_dir(samples_dir=samples_root)

    # 3 .txt files uploaded, .json ignored
    assert len(results) == 3
    uploaded_names = {p.name for (p, _, _) in results}
    assert uploaded_names == {
        "2026-04-30_br_Brazil_ChatForum.txt",
        "2026-04-30_br_1001631614451.txt",
        "2026-04-30_id_balichat.txt",
    }
    # All uploads went to the configured folder
    for call in files_api.create.call_args_list:
        assert call.kwargs["body"]["parents"] == ["FOLDER123"]


def test_upload_samples_dir_no_files(app_config, tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    results = du.upload_samples_dir(samples_dir=samples)
    assert results == []


def test_upload_samples_dir_swallows_auth_failure(app_config, tmp_path, monkeypatch):
    """If get_drive_service raises (e.g. token missing), we log and return []."""
    samples = tmp_path / "samples" / "br"
    samples.mkdir(parents=True)
    (samples / "f.txt").write_text("x")

    with patch.object(du, "get_drive_service", side_effect=FileNotFoundError("no token")):
        results = du.upload_samples_dir(samples_dir=tmp_path / "samples")
    assert results == []


# ---------------------------------------------------------------------------
# Credentials loading
# ---------------------------------------------------------------------------

def test_load_credentials_missing_token_raises(app_config, tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        du._load_credentials(missing)
