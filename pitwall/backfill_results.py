"""Fetch race classification (grid, finish, status) from the Ergast mirror.

Split out from ingestion on purpose. FastF1 spends Ergast calls during every
session load, and 186 races overruns the mirror's 500-calls-per-hour ceiling -
which cost 32 consecutive sessions on the first attempt and then stalled the
second in six-minute backoff waits.

Laps, weather, track status and race control need no Ergast at all; that was
verified directly, not assumed (see ``ingest.setup_fastf1``). Only the
classification does. So the bulk runs at full speed with Ergast disabled, and
this module fetches the small remainder at a deliberate pace: one call per
race, spaced to stay comfortably inside the budget.

What it recovers: starting grid position, finishing position, classification
status and points - the fields the counterfactual audit needs to say what a
team's strategy actually cost them.

Run with:
    python -m pitwall.backfill_results
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd

from pitwall import config
from pitwall.ingest import setup_fastf1

log = logging.getLogger("pitwall.backfill")

OUT_DIR = config.RAW / "classification"

# 450 calls/hour against a 500 ceiling leaves headroom for anything else on
# this machine touching the same mirror.
SECONDS_BETWEEN_CALLS = 8.0
MAX_ATTEMPTS = 3


def _needs_backfill(year: int, rnd: int) -> bool:
    """True if this race has no usable grid data yet."""
    if (OUT_DIR / f"{year}_{rnd:02d}.parquet").exists():
        return False
    existing = config.RAW / "results" / f"{year}_{rnd:02d}.parquet"
    if not existing.exists():
        return True
    try:
        r = pd.read_parquet(existing)
    except Exception:  # noqa: BLE001
        return True
    return "GridPosition" not in r.columns or r["GridPosition"].notna().sum() == 0


def fetch_one(year: int, rnd: int) -> pd.DataFrame | None:
    """Classification for one race, or None if the mirror has nothing."""
    from fastf1.ergast import Ergast

    erg = Ergast()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = erg.get_race_results(season=year, round=rnd)
            if not getattr(resp, "content", None):
                return None
            df = resp.content[0].copy()
            df["year"] = year
            df["round"] = rnd
            return df
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            if "RateLimitExceeded" in type(exc).__name__ or "500 calls/h" in str(exc):
                log.warning("rate limited at %s r%s; waiting 300s", year, rnd)
                time.sleep(300)
                continue
            log.warning("attempt %d/%d failed for %s r%s: %s", attempt, MAX_ATTEMPTS, year, rnd, msg)
            if attempt < MAX_ATTEMPTS:
                time.sleep(3.0 * attempt)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Backfill race classification from Ergast.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pace", type=float, default=SECONDS_BETWEEN_CALLS)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )
    setup_fastf1()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = config.RAW / "manifest.csv"
    if not manifest_path.exists():
        log.error("no manifest at %s; run pitwall.ingest first", manifest_path)
        return 1
    man = pd.read_csv(manifest_path)
    man = man[man["status"].isin(["ok", "cached"])]

    todo = [
        (int(r.year), int(r["round"]))
        for _, r in man.iterrows()
        if _needs_backfill(int(r.year), int(r["round"]))
    ]
    log.info("%d races need classification backfill (of %d ingested)", len(todo), len(man))

    n_ok = 0
    for i, (year, rnd) in enumerate(todo, 1):
        if args.limit and i > args.limit:
            break
        df = fetch_one(year, rnd)
        if df is None or df.empty:
            log.warning("[%3d/%d] %s r%-2s no classification returned", i, len(todo), year, rnd)
        else:
            df.to_parquet(OUT_DIR / f"{year}_{rnd:02d}.parquet", index=False)
            n_ok += 1
            log.info("[%3d/%d] %s r%-2s %d rows", i, len(todo), year, rnd, len(df))
        time.sleep(args.pace)

    log.info("backfilled %d/%d races -> %s", n_ok, len(todo), OUT_DIR)
    return 0


def load_classification() -> pd.DataFrame:
    """Every backfilled classification table, concatenated."""
    files = sorted(OUT_DIR.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())
