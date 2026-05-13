#!/usr/bin/env python3
"""
wikivoyage_batch.py — Nightly batch importer for WikiVoyage → Firestore.

Runs daily (via systemd timer at 12:00 UTC), picks the next N pending
countries from the priority list, imports each via wikivoyage_import,
and records progress in a state file.

State file: /home/teledigest/data/wikivoyage_batch_state.json
  {
    "th": {"status": "done", "patterns": 9649, "finished_at": "2026-05-11T..."},
    "fr": {"status": "pending"},
    ...
  }

Usage:
  # Dry run — show what would be imported today
  python -m teledigest.scripts.wikivoyage_batch --dry-run

  # Status report — see all countries and their state
  python -m teledigest.scripts.wikivoyage_batch --status

  # Run batch (default N=3)
  python -m teledigest.scripts.wikivoyage_batch

  # Override batch size
  python -m teledigest.scripts.wikivoyage_batch --n 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Country priority list — imported in this order (Tier 0 first = active chats)
PRIORITY = [
    # Tier 0 — active chats
    "th",
    "br",
    "ar",
    "id",
    "lk",
    "mu",
    "at",
    "be",
    "bg",
    "de",
    "vn",
    "fr",
    "ph",
    "tr",
    # Tier 1 — popular expat destinations
    "ae",
    "ge",
    "am",
    "az",
    "kz",
    "uz",
    "kg",
    "ua",
    "ru",
    "by",
    "es",
    "pt",
    "it",
    "gr",
    "hr",
    "rs",
    "me",
    "mk",
    "ba",
    "ro",
    "hu",
    "pl",
    "cz",
    "sk",
    "si",
    "ee",
    "lv",
    "lt",
    "fi",
    "se",
    "no",
    "dk",
    "nl",
    "ie",
    "gb",
    "ca",
    "us",
    "mx",
    "co",
    "pe",
    "cl",
    "uy",
    "py",
    "ec",
    "bo",
    "cr",
    "pa",
    "gt",
    "hn",
    "sv",
    "ni",
    "cu",
    "do",
    "ve",
    "jp",
    "kr",
    "cn",
    "tw",
    "sg",
    "my",
    "mm",
    "kh",
    "la",
    "mn",
    "np",
    "bd",
    "pk",
    "in",
    "il",
    "jo",
    "lb",
    "sa",
    "qa",
    "om",
    "cy",
    "ma",
    "tn",
    "eg",
    "dz",
    "ke",
    "tz",
    "ug",
    "gh",
    "ng",
    "sn",
    "et",
    "za",
    "zm",
    "zw",
    "mz",
    "mg",
    "na",
    "rw",
    "cm",
    "ci",
    "ly",
    "sd",
    "ye",
    "sy",
    "nz",
    # Tier 2
    "md",
    "xk",
    "li",
    "lu",
    "gt",
    "hn",
    "ni",
    "sv",
    "pa",
    "cu",
    "do",
    "ht",
    "cd",
    "ml",
    "mw",
    "ng",
    "dz",
    "ug",
    "ve",
    "qa",
]

# Remove duplicates while preserving order
seen: set[str] = set()
_deduped = []
for _c in PRIORITY:
    if _c not in seen:
        seen.add(_c)
        _deduped.append(_c)
PRIORITY = _deduped

DEFAULT_STATE_PATH = Path("/home/teledigest/data/wikivoyage_batch_state.json")
DEFAULT_CONFIG_PATH = Path("/config/teledigest.conf")
DEFAULT_BATCH_SIZE = 3

log = logging.getLogger("teledigest")


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(state: dict, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def mark_done(state: dict, country: str, patterns: int, state_path: Path) -> None:
    state[country] = {
        "status": "done",
        "patterns": patterns,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state, state_path)


def mark_failed(state: dict, country: str, error: str, state_path: Path) -> None:
    state[country] = {
        "status": "failed",
        "error": error,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state, state_path)


# ---------------------------------------------------------------------------
# Import one country (reuses wikivoyage_import internals)
# ---------------------------------------------------------------------------


def count_in_firestore(db, country: str, collection: str = "wikivoyage_base") -> int:
    """Count docs for a country in Firestore. Returns 0 if none or error."""
    try:
        docs = list(
            db.collection(collection).where("country", "==", country).limit(1).stream()
        )
        return len(docs)
    except Exception:
        return 0


def import_country(country: str, db, session) -> int:
    """Import one country. Returns total patterns in Firestore after import."""
    from teledigest.scripts.wikivoyage_import import (
        COUNTRY_WIKI_NAME,
        list_category_members,
        fetch_wikitext,
        parse_page,
        write_patterns,
    )

    wiki_name = COUNTRY_WIKI_NAME[country]
    log.info("=== WikiVoyage batch: starting %s (%s) ===", country, wiki_name)

    pages = list_category_members(session, wiki_name, max_depth=2)
    log.info("  %d pages found in Category:%s", len(pages), wiki_name)

    total_written = 0
    total_skipped = 0
    failed = []

    for i, title in enumerate(pages, 1):
        try:
            wt = fetch_wikitext(session, title)
            if not wt:
                continue
            patterns = parse_page(wt, country, title)
            written, skipped = write_patterns(db, country, title, patterns)
            total_written += written
            total_skipped += skipped
        except Exception as e:
            log.warning("  [%d/%d] %s failed: %s", i, len(pages), title, e)
            failed.append(title)

    log.info(
        "=== %s done: pages=%d wrote=%d skipped=%d failed=%d ===",
        country,
        len(pages),
        total_written,
        total_skipped,
        len(failed),
    )
    return total_written + total_skipped  # total patterns in DB


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------


def print_status(state: dict) -> None:
    from teledigest.scripts.wikivoyage_import import COUNTRY_WIKI_NAME

    print(f"\n{'Code':<6} {'Country':<30} {'Status':<10} {'Patterns':>10}  {'Date'}")
    print("-" * 75)
    done = pending = failed = 0
    for cc in PRIORITY:
        wiki = COUNTRY_WIKI_NAME.get(cc, "?")
        info = state.get(cc, {})
        status = info.get("status", "pending")
        patterns = info.get("patterns", "")
        date = (info.get("finished_at") or info.get("failed_at") or "")[:10]
        marker = "✓" if status == "done" else ("✗" if status == "failed" else "—")
        print(f"{cc:<6} {wiki:<30} {marker} {status:<8} {str(patterns):>10}  {date}")
        if status == "done":
            done += 1
        elif status == "failed":
            failed += 1
        else:
            pending += 1
    print("-" * 75)
    print(f"Done: {done}  Failed: {failed}  Pending: {pending}  Total: {len(PRIORITY)}")
    if pending > 0:
        days_left = (pending + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE
        print(f"ETA: ~{days_left} days at {DEFAULT_BATCH_SIZE} countries/day\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Countries per run (default {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--status", action="store_true", help="Show import status table and exit"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Scan Firestore and mark already-imported countries as done",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported, do not write",
    )
    parser.add_argument(
        "--reset",
        metavar="COUNTRY",
        help="Reset a country's state to pending (re-import it next run)",
    )
    parser.add_argument(
        "--state",
        default=str(DEFAULT_STATE_PATH),
        help=f"State file path (default {DEFAULT_STATE_PATH})",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Config path (default {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()

    # Init logging to both stderr and log file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(
                Path(args.state).parent / "wikivoyage_batch.log",
                encoding="utf-8",
            ),
        ],
    )

    state_path = Path(args.state)
    state = load_state(state_path)

    if args.reset:
        cc = args.reset.lower()
        if cc in state:
            del state[cc]
            save_state(state, state_path)
            print(f"Reset {cc} → pending")
        else:
            print(f"{cc} was already pending")
        return 0

    if args.status:
        print_status(state)
        return 0

    if args.sync:
        from pathlib import Path as _Path
        from teledigest.config import init_config
        from teledigest.scripts.wikivoyage_import import (
            _build_firestore_client,
            COUNTRY_WIKI_NAME,
        )
        from collections import Counter

        init_config(_Path(args.config))
        db = _build_firestore_client()
        print("Scanning Firestore wikivoyage_base...")
        docs = db.collection("wikivoyage_base").stream()
        counts = Counter(d.to_dict().get("country") for d in docs)
        synced = 0
        for cc, n in counts.items():
            if cc and state.get(cc, {}).get("status") != "done":
                mark_done(state, cc, n, state_path)
                print(f"  {cc}: marked done ({n} docs)")
                synced += 1
        print(f"Synced {synced} countries from Firestore.")
        print_status(state)
        return 0

    # Pick next N pending countries
    from teledigest.scripts.wikivoyage_import import COUNTRY_WIKI_NAME

    queue = [
        cc
        for cc in PRIORITY
        if cc in COUNTRY_WIKI_NAME and state.get(cc, {}).get("status") != "done"
    ]

    if not queue:
        log.info("WikiVoyage batch: all countries done!")
        return 0

    batch = queue[: args.n]

    if args.dry_run:
        print(f"Would import {len(batch)} countries: {', '.join(batch)}")
        print(f"Remaining after this run: {len(queue) - len(batch)}")
        return 0

    log.info(
        "WikiVoyage batch: importing %d countries: %s", len(batch), ", ".join(batch)
    )

    from pathlib import Path as _Path
    from teledigest.config import init_config
    from teledigest.scripts.wikivoyage_import import _build_firestore_client
    import requests

    init_config(_Path(args.config))
    db = _build_firestore_client()
    session = requests.Session()
    session.headers["User-Agent"] = "teledigest-wikivoyage-bot/1.0 (teledigest project)"

    for i, cc in enumerate(batch):
        if i > 0:
            log.info(
                "Pausing 60s between countries to respect WikiVoyage rate limits..."
            )
            time.sleep(60)
        try:
            # Check Firestore first — skip if already has data (imported manually)
            existing = count_in_firestore(db, cc)
            if existing > 0:
                log.info(
                    "Country %s already has data in Firestore (%d docs) — marking done",
                    cc,
                    existing,
                )
                mark_done(state, cc, existing, state_path)
                continue
            patterns = import_country(cc, db, session)
            mark_done(state, cc, patterns, state_path)
        except Exception as e:
            log.error("Country %s failed: %s", cc, e)
            mark_failed(state, cc, str(e), state_path)

    log.info("WikiVoyage batch run complete. Imported: %s", ", ".join(batch))
    remaining = len([c for c in queue[args.n :] if COUNTRY_WIKI_NAME.get(c)])
    log.info("Remaining in queue: %d countries", remaining)
    return 0


if __name__ == "__main__":
    sys.exit(main())
