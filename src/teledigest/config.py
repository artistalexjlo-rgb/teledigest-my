from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional

from platformdirs import user_config_dir

APP_NAME = "teledigest"
APP_AUTHOR = "Igor Opaniuk"  # optional, used on Windows


@dataclass
class TelegramConfig:
    api_id: int
    api_hash: str
    bot_token: str
    sessions_dir: Path = Path("data")


@dataclass
class SourceChannel:
    """A single Telegram source channel/chat."""

    name: str
    url: str
    country: str
    language: str = "ru"


@dataclass
class SourcesConfig:
    """Multi-country source configuration."""

    channels: List[SourceChannel] = field(default_factory=list)
    digest_targets: Dict[str, str] = field(default_factory=dict)

    def countries(self) -> list[str]:
        """List of unique country codes."""
        return list(dict.fromkeys(ch.country for ch in self.channels))

    def channels_for_country(self, country: str) -> list[SourceChannel]:
        return [ch for ch in self.channels if ch.country == country]

    def channel_urls(self) -> list[str]:
        """All channel URLs (for Telegram joining)."""
        return [ch.url for ch in self.channels]


_DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that summarizes Telegram messages into a concise digest."""
_DEFAULT_USER_PROMPT = (
    """Summarize the following Telegram messages for {DAY}:\n\n{MESSAGES}"""
)
_DEFAULT_SYSTEM_BRIEF_PROMPT = """You are a Telegram digest bot. Create a very concise summary suitable for a single Telegram message."""
_DEFAULT_USER_BRIEF_PROMPT = (
    "Create a brief summary (max 4000 characters including HTML)"
    " of the following digest:\n\n{DIGEST}"
)


@dataclass
class ExtractionLLMConfig:
    """Separate LLM config for heavy Q&A extraction (e.g. free Yandex tier)."""

    model: str = ""
    api_key: str = ""
    base_url: Optional[str] = None
    temperature: float = 0.2


@dataclass
class LLMConfig:
    model: str
    api_key: str
    system_prompt: str
    user_prompt: str
    base_url: Optional[str] = None
    temperature: float = 0.4
    max_messages: int = 1000
    system_brief_prompt: str = field(
        default_factory=lambda: _DEFAULT_SYSTEM_BRIEF_PROMPT
    )
    user_brief_prompt: str = field(default_factory=lambda: _DEFAULT_USER_BRIEF_PROMPT)
    extraction: ExtractionLLMConfig = field(default_factory=ExtractionLLMConfig)


@dataclass
class TelegraphConfig:
    author_name: str = "TeleDigest"
    author_url: str = ""
    access_token: Optional[str] = None


@dataclass
class BotConfig:
    channels: List[str]
    summary_target: str
    time_zone: str = "Europe/Warsaw"
    summary_hour: int = 21
    summary_minute: int = 0
    summary_brief: bool = False
    allowed_users_raw: str = ""  # e.g. "@user1,12345678"
    bot_name: str = "МОЗГ"
    blocked_senders: frozenset = frozenset()  # bot usernames/IDs to ignore

    def _raw_parts(self) -> List[str]:
        return [x.strip() for x in self.allowed_users_raw.split(",") if x.strip()]

    @cached_property
    def allowed_user_ids(self) -> frozenset:
        result: set[int] = set()
        for x in self._raw_parts():
            if not x.startswith("@"):
                try:
                    result.add(int(x))
                except ValueError:
                    log.warning("Invalid user ID in allowed_users: %r", x)
        return frozenset(result)

    @cached_property
    def allowed_user_names(self) -> frozenset:
        return frozenset(
            x.lstrip("@").lower() for x in self._raw_parts() if x.startswith("@")
        )


@dataclass
class StorageConfig:
    rag_keywords: list[str]
    db_path: Path = Path("messages_fts.db")


@dataclass
class GoogleConfig:
    """Google Drive + Firestore settings.

    Drive: OAuth user token (token_path) — SA can't create files in personal Drive.
    Firestore: Service Account (service_account_path) — no expiry, no scope drift.
    """

    drive_folder_id: str = ""
    token_path: Path = Path("google-token.json")  # Drive OAuth
    service_account_path: Path = Path("service-account.json")  # Firestore SA
    enabled: bool = False
    # Firestore (for channel poster reading telegram_queue collection)
    firestore_project_id: str = ""
    firestore_database: str = "default"
    firestore_collection: str = "telegram_queue"
    # Firestore collection for МОЗГ assistant data (wisdom_base)
    assistant_collection: str = "wisdom_base"


@dataclass
class GeminiConfig:
    """Gemini API settings for МОЗГ chat assistant."""

    api_key: str = ""
    # Legacy synchronous model (used by Apps Script extraction-style flows
    # and as fallback if Live API errors). Shares 500 RPD on free tier.
    model: str = "gemini-3.1-flash-lite-preview"
    # Gemini Live API model — bidirectional streaming session. Free tier
    # quota is "Unlimited RPD" on Live, so МОЗГ answers don't compete
    # with extraction for the per-day request budget. Verify the exact
    # current ID in AI Studio (https://aistudio.google.com) — Google
    # rotates preview model IDs roughly monthly.
    live_model: str = "gemini-3.1-flash-live-preview"
    enabled: bool = False


@dataclass
class ChannelConfig:
    """Auto-poster: reads stories from Firestore, posts to a Telegram channel.

    Uses the same OAuth user creds as Drive (token.json must include
    `datastore` scope alongside `drive.file`).
    """

    target: str = ""  # @luky_channel or numeric chat_id (e.g. -100...)
    posts_per_day: int = 5
    window_start_hour: int = 8  # 08:00
    window_end_hour: int = 24  # exclusive — 24 = up to 23:59:59
    jitter_minutes: int = 5  # ± random minutes per slot to look natural
    enabled: bool = False
    # Optional comma-separated list of country codes to exclude from posting
    exclude_countries: str = ""


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class AppConfig:
    telegram: TelegramConfig
    bot: BotConfig
    llm: LLMConfig
    storage: StorageConfig = field(default_factory=lambda: StorageConfig([]))
    logging: LoggingConfig = field(default_factory=lambda: LoggingConfig())
    telegraph: TelegraphConfig = field(default_factory=TelegraphConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    google: GoogleConfig = field(default_factory=GoogleConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)


_CONFIG: Optional[AppConfig] = None

log = logging.getLogger(APP_NAME)


def _default_config_path() -> Path:
    """
    Determine the default config file path in a cross-platform way.
    Example:
      - Linux:  ~/.config/teledigest/config.toml
      - macOS:  ~/Library/Application Support/teledigest/config.toml
      - Win:    %APPDATA%\\teledigest\\config.toml
    """
    config_dir = Path(user_config_dir(APP_NAME, APP_AUTHOR))
    return config_dir / "config.toml"


def _load_toml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("rb") as f:
        data = tomllib.load(f)
    return data


def _locate_config_path(
    explicit_path: Optional[Path] = None,
    create_parent: bool = False,
) -> Path:
    """
    Decide which config path to use.

    Precedence:
      1. explicit_path (e.g. from CLI)
      2. TELEGRAM_DIGEST_CONFIG env var
      3. OS-specific default user config path
    """
    if explicit_path is not None:
        config_path = explicit_path.expanduser()
    else:
        config_path = _default_config_path()

    if create_parent:
        config_path.parent.mkdir(parents=True, exist_ok=True)

    return config_path


def _parse_telegram(raw: Dict[str, Any]) -> TelegramConfig:
    tg_raw = raw.get("telegram") or {}
    try:
        return TelegramConfig(
            api_id=int(os.environ.get("TELEGRAM_API_ID") or tg_raw["api_id"]),
            api_hash=str(os.environ.get("TELEGRAM_API_HASH") or tg_raw["api_hash"]),
            bot_token=str(os.environ.get("TELEGRAM_BOT_TOKEN") or tg_raw["bot_token"]),
            sessions_dir=Path(tg_raw.get("sessions_dir", "data")),
        )
    except KeyError as e:
        raise KeyError(f"Missing required [telegram] field in config: {e!s}") from e


def _parse_bot(raw: Dict[str, Any]) -> BotConfig:
    bot_raw = raw.get("bot") or {}
    channels = bot_raw.get("channels") or []
    if not isinstance(channels, list) or not channels:
        raise ValueError("Config [bot].channels must be a non-empty list.")

    bot = BotConfig(
        channels=[str(c).strip() for c in channels],
        summary_target=str(bot_raw.get("summary_target", "")).strip(),
        summary_hour=int(bot_raw.get("summary_hour", 21)),
        summary_minute=int(bot_raw.get("summary_minute", 0)),
        summary_brief=bool(bot_raw.get("summary_brief", False)),
        allowed_users_raw=str(bot_raw.get("allowed_users", "")),
        time_zone=str(bot_raw.get("time_zone", "Europe/Warsaw")),
        bot_name=str(bot_raw.get("bot_name", "МОЗГ")),
        blocked_senders=frozenset(
            str(s).strip().lower() for s in (bot_raw.get("blocked_senders") or [])
        ),
    )

    if not bot.summary_target:
        raise ValueError("Config [bot].summary_target is required.")
    if not (0 <= bot.summary_hour <= 23):
        raise ValueError("Config [bot].summary_hour must be between 0 and 23.")
    if not (0 <= bot.summary_minute <= 59):
        raise ValueError("Config [bot].summary_minute must be between 0 and 59.")

    return bot


def _parse_storage(raw: Dict[str, Any]) -> StorageConfig:
    storage_raw = raw.get("storage") or {}
    rag_raw = storage_raw.get("rag") or {}
    db_path_str = storage_raw.get("db_path", "/data/messages_fts.db")
    keywords = rag_raw.get("keywords") or []
    return StorageConfig(db_path=Path(db_path_str), rag_keywords=keywords)


def _parse_llm(raw: Dict[str, Any]) -> LLMConfig:
    llm_raw = raw.get("llm") or {}
    prompts_raw = llm_raw.get("prompts") or {}
    llm = LLMConfig(
        api_key=str(os.environ.get("DEEPSEEK_API_KEY") or llm_raw.get("api_key", "")),
        model=str(llm_raw.get("model", "gpt-5.1")),
        system_prompt=str(prompts_raw.get("system", _DEFAULT_SYSTEM_PROMPT)),
        user_prompt=str(prompts_raw.get("user", _DEFAULT_USER_PROMPT)),
        max_messages=int(prompts_raw.get("max_messages", 1000)),
        system_brief_prompt=str(
            prompts_raw.get("system_brief", _DEFAULT_SYSTEM_BRIEF_PROMPT)
        ),
        user_brief_prompt=str(
            prompts_raw.get("user_brief", _DEFAULT_USER_BRIEF_PROMPT)
        ),
        base_url=str(llm_raw.get("base_url", "")) or None,
        temperature=float(llm_raw.get("temperature", 0.4)),
    )

    if not (0.0 <= llm.temperature <= 2.0):
        raise ValueError("Config [llm].temperature must be between 0.0 and 2.0.")
    if not llm.api_key:
        raise ValueError("Config [llm].api_key is required.")

    # Parse optional [llm.extraction] for separate extraction LLM
    ext_raw = llm_raw.get("extraction") or {}
    if ext_raw.get("api_key"):
        llm.extraction = ExtractionLLMConfig(
            model=str(ext_raw.get("model", "")),
            api_key=str(ext_raw.get("api_key", "")),
            base_url=str(ext_raw.get("base_url", "")) or None,
            temperature=float(ext_raw.get("temperature", 0.2)),
        )

    return llm


def _parse_logging(raw: Dict[str, Any]) -> LoggingConfig:
    logging_raw = raw.get("logging") or {}
    level = str(logging_raw.get("level", "INFO"))
    if not isinstance(getattr(logging, level.upper(), None), int):
        raise ValueError(
            f"Config [logging].level is invalid: {level!r}. "
            "Valid values are: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    return LoggingConfig(level=level)


def _parse_telegraph(raw: Dict[str, Any]) -> TelegraphConfig:
    tph_raw = raw.get("telegraph") or {}
    return TelegraphConfig(
        author_name=str(tph_raw.get("author_name", "TeleDigest")),
        author_url=str(tph_raw.get("author_url", "")),
        access_token=(
            str(tph_raw["access_token"]) if tph_raw.get("access_token") else None
        ),
    )


def _parse_sources(raw: Dict[str, Any]) -> SourcesConfig:
    sources_raw = raw.get("sources") or {}
    channels_raw = sources_raw.get("channels") or []
    targets_raw = sources_raw.get("digest_targets") or {}

    channels = []
    for ch in channels_raw:
        channels.append(
            SourceChannel(
                name=str(ch.get("name", "")),
                url=str(ch.get("url", "")),
                country=str(ch.get("country", "")).lower(),
                language=str(ch.get("language", "ru")),
            )
        )

    digest_targets = {str(k).lower(): str(v) for k, v in targets_raw.items()}

    return SourcesConfig(channels=channels, digest_targets=digest_targets)


def _parse_google(raw: Dict[str, Any]) -> GoogleConfig:
    g_raw = raw.get("google") or {}
    folder_id = str(g_raw.get("drive_folder_id", "")).strip()
    return GoogleConfig(
        drive_folder_id=folder_id,
        token_path=Path(g_raw.get("token_path", "google-token.json")),
        service_account_path=Path(
            g_raw.get("service_account_path", "service-account.json")
        ),
        enabled=bool(folder_id) and bool(g_raw.get("enabled", True)),
        firestore_project_id=str(g_raw.get("firestore_project_id", "")).strip(),
        firestore_database=str(g_raw.get("firestore_database", "default")).strip()
        or "default",
        firestore_collection=str(
            g_raw.get("firestore_collection", "telegram_queue")
        ).strip()
        or "telegram_queue",
        assistant_collection=str(
            g_raw.get("assistant_collection", "wisdom_base")
        ).strip()
        or "wisdom_base",
    )


def _parse_gemini(raw: Dict[str, Any]) -> GeminiConfig:
    g_raw = raw.get("gemini") or {}
    api_key = str(os.environ.get("GEMINI_API_KEY") or g_raw.get("api_key", "")).strip()
    model = (
        str(g_raw.get("model", "gemini-3.1-flash-lite-preview")).strip()
        or "gemini-3.1-flash-lite-preview"
    )
    live_model = (
        str(g_raw.get("live_model", "gemini-3.1-flash-live-preview")).strip()
        or "gemini-3.1-flash-live-preview"
    )
    return GeminiConfig(
        api_key=api_key,
        model=model,
        live_model=live_model,
        enabled=bool(api_key),
    )


def _parse_channel(raw: Dict[str, Any]) -> ChannelConfig:
    c_raw = raw.get("channel") or {}
    target = str(c_raw.get("target", "")).strip()
    return ChannelConfig(
        target=target,
        posts_per_day=int(c_raw.get("posts_per_day", 5)),
        window_start_hour=int(c_raw.get("window_start_hour", 8)),
        window_end_hour=int(c_raw.get("window_end_hour", 24)),
        jitter_minutes=int(c_raw.get("jitter_minutes", 5)),
        enabled=bool(target) and bool(c_raw.get("enabled", True)),
        exclude_countries=str(c_raw.get("exclude_countries", "")).strip(),
    )


def _parse_app_config(raw: Dict[str, Any]) -> AppConfig:
    """
    Convert the raw TOML dict into typed AppConfig.
    Raises KeyError/ValueError if required sections/fields are missing or invalid.
    """
    sources = _parse_sources(raw)

    return AppConfig(
        telegram=_parse_telegram(raw),
        bot=_parse_bot(raw),
        llm=_parse_llm(raw),
        storage=_parse_storage(raw),
        logging=_parse_logging(raw),
        telegraph=_parse_telegraph(raw),
        sources=sources,
        google=_parse_google(raw),
        channel=_parse_channel(raw),
        gemini=_parse_gemini(raw),
    )


def init_config(
    explicit_path: Optional[Path] = None,
) -> AppConfig:

    global _CONFIG
    if _CONFIG is not None:
        if explicit_path is not None:
            log.warning(
                "init_config() called again with explicit_path=%s, "
                "but config is already loaded; the new path will be ignored.",
                explicit_path,
            )
        return _CONFIG

    config_path = _locate_config_path(explicit_path)

    log.debug("Loading config from %s", config_path)
    raw = _load_toml(config_path)
    _CONFIG = _parse_app_config(raw)
    _configure_logging(_CONFIG.logging)

    return _CONFIG


def get_config() -> AppConfig:
    """
    Get the global AppConfig.
    Raises if init_config() hasn't been called yet.
    """
    if _CONFIG is None:
        raise RuntimeError("Config not initialized. Call init_config() first.")
    return _CONFIG


def _configure_logging(logging_cfg: LoggingConfig) -> None:
    """
    Basic logging setup based on config.
    """
    level = getattr(logging, logging_cfg.level.upper())
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    log.info("Logging configured at %s level", logging_cfg.level.upper())
