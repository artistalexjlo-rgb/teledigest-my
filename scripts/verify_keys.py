#!/usr/bin/env python3
"""verify_keys.py — ping every key in GEMINI_API_KEYS and print which work.

Run before bulk migration to catch revoked / typo'd / exhausted keys.

Each key gets ONE tiny embed call (~32 chars) to test that:
  - the key is valid (no 400/401/403)
  - it has quota for at least one call right now (no 429)

Output:
  key #0  prefix=AIza...xyz   OK         dim=1536
  key #2  prefix=AIza...abc   QUOTA      RESOURCE_EXHAUSTED
  key #3  prefix=AIza...def   INVALID    API_KEY_INVALID

Usage inside the container:
    python /app/scripts/verify_keys.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="/config/teledigest.conf")
    args = parser.parse_args()

    from teledigest.config import init_config

    init_config(Path(args.config))

    from google import genai
    from google.genai import types as genai_types

    from teledigest.gemini_brain import (
        _EMBEDDING_DIM_V2,
        _EMBEDDING_MODEL_V2,
        _get_embedding_api_keys,
    )

    keys = _get_embedding_api_keys()
    if not keys:
        print("No keys configured (set GEMINI_API_KEYS or GEMINI_API_KEY).")
        return 1
    print(f"Found {len(keys)} key(s). Pinging each...\n")

    cfg = genai_types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=_EMBEDDING_DIM_V2,
    )

    ok_count = 0
    for i, key in enumerate(keys):
        # Hide all but last 4 chars.
        masked = (key[:4] + "..." + key[-4:]) if len(key) > 12 else "(short)"
        try:
            client = genai.Client(api_key=key)
            res = client.models.embed_content(
                model=_EMBEDDING_MODEL_V2,
                contents="ping",
                config=cfg,
            )
            vecs = res.embeddings or []
            if vecs and vecs[0].values:
                dim = len(vecs[0].values)
                print(f"  key #{i}  prefix={masked}   OK         dim={dim}")
                ok_count += 1
            else:
                print(f"  key #{i}  prefix={masked}   EMPTY      no vector returned")
        except Exception as e:
            err = str(e).split("\n")[0][:120]
            status = "ERROR"
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                status = "QUOTA"
            elif (
                "400" in err or "401" in err or "403" in err or "API_KEY_INVALID" in err
            ):
                status = "INVALID"
            print(f"  key #{i}  prefix={masked}   {status:10s} {err}")

    print(f"\n{ok_count}/{len(keys)} keys healthy.")
    return 0 if ok_count == len(keys) else 1


if __name__ == "__main__":
    # Silence noisy SDK INFO logs for clean output.
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)
    sys.exit(main())
