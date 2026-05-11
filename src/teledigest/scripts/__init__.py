"""One-off / scheduled scripts that piggyback on the teledigest package.

Each module here is meant to be invoked as
``python -m teledigest.scripts.<module> [--args]`` either ad-hoc or from a
systemd timer on the host (via ``docker exec``). They reuse the package's
config, db, and daily_samples helpers so they don't drift out of sync with
the runtime.
"""
