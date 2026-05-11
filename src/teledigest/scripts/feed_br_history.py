#!/usr/bin/env python3
"""
feed_br_history.py — Slowly feed the year of scraped BR messages into the
Apps Script mining pipeline.

Why:
    The bot only scraped BR (Brazil) for a long time before the multi-country
    setup. That archive sits in messages table but never went through the
    daily_samples → Drive → Apps Script → Gemini → Firestore pipeline.
    Reprocessing it gives the assistant a much richer wisdom_base for BR.

How it works:
    1. Find all distinct (date, source) pairs in messages for the target
       country.
    2. Skip pairs that already have a sample file on disk (covers re-runs
       and avoids clobbering daily samples that have already been written
       for recent days).
    3. Sort oldest-first and take the next N pairs.
    4. Dump each pair to samples/{country}/{date}_{country}_{slug}.txt.
       If body is over CHAR_CAP characters (~25K Gemini tokens), split into
       _part01.txt, _part02.txt, ... so individual requests stay well under
       the model's TPM and per-request limits.
    5. drive_uploader (separate scheduled job) syncs the new files to Drive.
    6. Apps Script picks them up on its 15-min trigger.

Designed for unattended daily systemd runs: 15 pairs/day means a 365-day
archive drains in ~24 days while leaving daily-scraped files from active
countries plenty of headroom in the 500 RPD Gemini free-tier budget.

Usage (inside container or anywhere teledigest is installed):
    python -m teledigest.scripts.feed_br_history                   # 15 pairs
    python -m teledigest.scripts.feed_br_history --batch 20        # 20 pairs
    python -m teledigest.scripts.feed_br_history --country br --batch 10
    python -m teledigest.scripts.feed_br_history --dry-run         # show, no write
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from teledigest.config import log
from teledigest.daily_samples import (
    SampleTarget,
    _channel_slug,
    _fetch_messages,
    _format_line,
    get_sample_targets,
    get_samples_dir,
)


# 80K chars ≈ 25-30K tokens, safely under 3.1-flash-lite-preview's 250K TPM
# and well within any per-request size limit. Headroom for retries.
CHAR_CAP = 80_000


def list_message_dates(country: str, channel: str) -> list[dt.date]:
    """All distinct dates with at least one human message for this source."""
    from teledigest.db import get_db_connection

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT substr(date, 1, 10) AS d
            FROM messages
            WHERE country = ? AND channel = ?
              AND (is_bot = 0 OR is_bot IS NULL)
              AND text IS NOT NULL AND length(text) > 0
            ORDER BY d ASC
            """,
            (country, channel),
        )
        return [dt.date.fromisoformat(row[0]) for row in cur.fetchall()]


def is_already_dumped(samples_dir: Path, target: SampleTarget, day: dt.date) -> bool:
    """Bare file or any chunked _partNN file present → consider done."""
    country_dir = samples_dir / target.country
    slug = _channel_slug(target.channel)
    base = f"{day.isoformat()}_{target.country}_{slug}"
    if (country_dir / f"{base}.txt").exists():
        return True
    if any(country_dir.glob(f"{base}_part*.txt")):
        return True
    return False


def dump_chunked(target: SampleTarget, day: dt.date, samples_dir: Path) -> list[Path]:
    """Write a day's messages, splitting into _partNN files past CHAR_CAP."""
    rows = _fetch_messages(target.country, target.channel, day)
    if not rows:
        return []

    country_dir = samples_dir / target.country
    country_dir.mkdir(parents=True, exist_ok=True)
    slug = _channel_slug(target.channel)
    base = f"{day.isoformat()}_{target.country}_{slug}"

    # Accumulate lines into chunks. Start a new chunk whenever adding the
    # next line would push the running size past CHAR_CAP — but only if the
    # current chunk has at least one line, otherwise a single oversized line
    # would loop forever.
    chunks: list[list[str]] = [[]]
    sizes: list[int] = [0]
    for (_id, date_iso, text, sender_id, reply_to_msg_id) in rows:
        line = _format_line(date_iso, text, sender_id, reply_to_msg_id)
        addition = len(line) + 1  # +1 for trailing newline
        if sizes[-1] + addition > CHAR_CAP and chunks[-1]:
            chunks.append([])
            sizes.append(0)
        chunks[-1].append(line)
        sizes[-1] += addition

    written: list[Path] = []
    multi = len(chunks) > 1
    for idx, body_lines in enumerate(chunks, start=1):
        name = f"{base}_part{idx:02d}.txt" if multi else f"{base}.txt"
        header = (
            f"# country={target.country} channel={target.channel} "
            f"day={day.isoformat()} (UTC times) messages={len(body_lines)}"
        )
        if multi:
            header += f" chunk={idx}/{len(chunks)}"
        header += "\n"
        path = country_dir / name
        path.write_text(header + "\n".join(body_lines) + "\n", encoding="utf-8")
        written.append(path)

    return written


def find_pending_pairs(country: str, samples_dir: Path) -> list[tuple[SampleTarget, dt.date]]:
    """List (target, day) pairs for country that aren't on disk yet."""
    pending: list[tuple[SampleTarget, dt.date]] = []
    for target in get_sample_targets():
        if target.country != country:
            continue
        for day in list_message_dates(target.country, target.channel):
            if not is_already_dumped(samples_dir, target, day):
                pending.append((target, day))
    pending.sort(key=lambda p: p[1])
    return pending


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--country", default="br",
                        help="ISO country code (default: br)")
    parser.add_argument("--batch", type=int, default=15,
                        help="How many (source, day) pairs to dump this run "
                             "(default: 15, sized to fit Gemini free-tier RPD)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be dumped, write nothing")
    args = parser.parse_args()

    samples_dir = get_samples_dir()
    pending = find_pending_pairs(args.country, samples_dir)

    if not pending:
        log.info("Country=%s: no pending pairs — archive fully fed.", args.country)
        return 0

    take = pending[: args.batch]
    log.info(
        "Country=%s: %d pending pairs total, processing %d this run (oldest first).",
        args.country, len(pending), len(take),
    )

    if args.dry_run:
        for target, day in take:
            print(f"WOULD DUMP: {day.isoformat()}  {target.channel}")
        return 0

    dumped_files = 0
    for target, day in take:
        try:
            written = dump_chunked(target, day, samples_dir)
            log.info(
                "Dumped country=%s channel=%s day=%s -> %d file(s) %s",
                target.country, target.channel, day.isoformat(),
                len(written), [p.name for p in written],
            )
            dumped_files += len(written)
        except Exception as e:
            log.exception("Failed to dump %s %s: %s", target.channel, day, e)

    remaining = len(pending) - len(take)
    eta_runs = (remaining + args.batch - 1) // args.batch if args.batch else 0
    log.info(
        "Run complete: %d files written from %d pairs. "
        "%d pairs remaining (~%d more runs at batch=%d).",
        dumped_files, len(take), remaining, eta_runs, args.batch,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
