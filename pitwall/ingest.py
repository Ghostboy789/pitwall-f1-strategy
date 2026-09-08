"""Phase 1 - cached ingestion of every race session in scope.

One session per file, so a run is resumable: if the process dies at race 140
of 180, re-running it picks up where it stopped instead of re-downloading a
gigabyte. Every session's outcome - including every failure - is recorded in
a manifest, because the failures are themselves a result (FastF1's timing
archive has real holes, and the methodology has to say which).

Run with:
    python -m pitwall.ingest
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pitwall import config

log = logging.getLogger("pitwall.ingest")

# Sub-directories of data/raw, one table per kind.
KINDS = ("laps", "weather", "track_status", "results", "race_control")

# Be polite to the Ergast mirror FastF1 falls back on for first-lap times.
SLEEP_BETWEEN_SESSIONS_S = 1.0
MAX_ATTEMPTS = 3


def setup_fastf1() -> None:
    """Point FastF1 at the project cache and quieten its logger."""
    import fastf1

    fastf1.Cache.enable_cache(str(config.CACHE))
    fastf1.set_log_level("ERROR")


def timedeltas_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Replace timedelta columns with float seconds.

    Parquet round-trips timedeltas, but every consumer downstream wants
    seconds, and converting once here keeps that conversion out of the
    modelling code.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_timedelta64_dtype(out[col]):
            out[col] = out[col].dt.total_seconds()
    return out


@dataclass
class SessionRecord:
    """One row of the ingestion manifest."""

    year: int
    round: int
    event_name: str
    location: str
    country: str
    event_date: str
    event_format: str
    status: str = "pending"
    error: str = ""
    n_laps: int = 0
    n_drivers: int = 0
    n_weather: int = 0
    n_track_status: int = 0
    n_results: int = 0
    n_race_control: int = 0
    compounds: str = ""
    total_laps: int = 0
    has_laptime: int = 0
    has_compound: int = 0
    has_tyrelife: int = 0
    attempts: int = 0
    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def enumerate_races(seasons=config.SEASONS) -> pd.DataFrame:
    """List every completed championship race in the given seasons.

    Excludes pre-season testing and any event whose date is still in the
    future, since a race that has not happened has no timing data.
    """
    import fastf1

    setup_fastf1()
    frames = []
    today = pd.Timestamp.now().normalize()

    for year in seasons:
        try:
            sched = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:  # noqa: BLE001 - a missing season must not kill the run
            log.error("schedule %s failed: %s", year, exc)
            continue
        sched = sched[sched["RoundNumber"] >= 1]
        sched = sched[pd.to_datetime(sched["EventDate"]) <= today]
        frames.append(
            pd.DataFrame(
                {
                    "year": year,
                    "round": sched["RoundNumber"].astype(int),
                    "event_name": sched["EventName"],
                    "location": sched["Location"],
                    "country": sched["Country"],
                    "event_date": pd.to_datetime(sched["EventDate"]).dt.strftime("%Y-%m-%d"),
                    "event_format": sched["EventFormat"],
                }
            )
        )

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values(["year", "round"])
    return out.reset_index(drop=True)


def paths_for(year: int, rnd: int) -> dict[str, Path]:
    """Parquet destination for each table kind of one session."""
    return {k: config.RAW / k / f"{year}_{rnd:02d}.parquet" for k in KINDS}


def ingest_session(rec: SessionRecord, force: bool = False) -> SessionRecord:
    """Load one race and write its tables. Never raises."""
    import fastf1

    paths = paths_for(rec.year, rec.round)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    if not force and paths["laps"].exists():
        rec.status = "cached"
        try:
            laps = pd.read_parquet(paths["laps"])
            rec.n_laps = len(laps)
            rec.n_drivers = int(laps["Driver"].nunique()) if "Driver" in laps else 0
        except Exception:  # noqa: BLE001
            pass
        return rec

    last_exc = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rec.attempts = attempt
        try:
            ses = fastf1.get_session(rec.year, rec.round, "R")
            ses.load(laps=True, telemetry=False, weather=True, messages=True)

            # `.laps` raises DataNotLoadedError when the underlying timing
            # stream failed to parse. That is a hole in the source archive,
            # not a bug here, and it must be recorded rather than swallowed.
            laps = ses.laps
            if laps is None or len(laps) == 0:
                raise ValueError("session loaded but lap table is empty")

            laps = timedeltas_to_seconds(pd.DataFrame(laps))
            laps["year"] = rec.year
            laps["round"] = rec.round
            laps["event_name"] = rec.event_name
            laps["location"] = rec.location
            laps["country"] = rec.country
            laps["event_date"] = rec.event_date
            laps["event_format"] = rec.event_format
            laps["era"] = config.ERAS.get(rec.year, "unknown")
            laps.to_parquet(paths["laps"], index=False)

            rec.n_laps = len(laps)
            rec.n_drivers = int(laps["Driver"].nunique())
            rec.total_laps = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max() or 0)
            rec.has_laptime = int(laps["LapTime"].notna().sum())
            rec.has_compound = int(laps["Compound"].notna().sum())
            rec.has_tyrelife = int(laps["TyreLife"].notna().sum())
            comps = sorted(
                {
                    str(c)
                    for c in laps["Compound"].dropna().unique()
                    if str(c).upper() not in ("NAN", "NONE", "")
                }
            )
            rec.compounds = "|".join(comps)

            # Ancillary tables. A failure here must not lose the laps.
            getters = {
                "weather": lambda: ses.weather_data,
                "track_status": lambda: ses.track_status,
                "results": lambda: ses.results,
                "race_control": lambda: ses.race_control_messages,
            }
            for kind, getter in getters.items():
                try:
                    tbl = getter()
                    if tbl is None or len(tbl) == 0:
                        continue
                    tbl = timedeltas_to_seconds(pd.DataFrame(tbl))
                    tbl["year"] = rec.year
                    tbl["round"] = rec.round
                    tbl.to_parquet(paths[kind], index=False)
                    setattr(rec, f"n_{kind}", len(tbl))
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "%s r%s %s: %s table failed: %s",
                        rec.year, rec.round, rec.event_name, kind, exc,
                    )

            rec.status = "ok"
            rec.error = ""
            return rec

        except Exception as exc:  # noqa: BLE001
            last_exc = f"{type(exc).__name__}: {exc}"
            log.warning(
                "attempt %s/%s failed for %s r%s %s: %s",
                attempt, MAX_ATTEMPTS, rec.year, rec.round, rec.event_name, last_exc,
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(2.0 * attempt)

    rec.status = "failed"
    rec.error = last_exc[:400]
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest F1 race sessions into parquet.")
    ap.add_argument("--seasons", type=int, nargs="*", default=list(config.SEASONS))
    ap.add_argument("--force", action="store_true", help="re-download sessions already on disk")
    ap.add_argument("--limit", type=int, default=0, help="stop after N sessions (smoke test)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    setup_fastf1()

    races = enumerate_races(tuple(args.seasons))
    log.info("scope: %d races across seasons %s", len(races), args.seasons)

    manifest_path = config.RAW / "manifest.csv"
    records: list[dict] = []
    if manifest_path.exists() and not args.force:
        records = pd.read_csv(manifest_path).fillna("").to_dict("records")
    done = {
        (int(r["year"]), int(r["round"]))
        for r in records
        if str(r.get("status")) in ("ok", "failed", "cached")
    }

    n = 0
    for row in races.itertuples(index=False):
        if args.limit and n >= args.limit:
            break
        if (int(row.year), int(row.round)) in done and not args.force:
            continue
        rec = SessionRecord(
            year=int(row.year),
            round=int(row.round),
            event_name=str(row.event_name),
            location=str(row.location),
            country=str(row.country),
            event_date=str(row.event_date),
            event_format=str(row.event_format),
        )
        t0 = time.time()
        rec = ingest_session(rec, force=args.force)
        n += 1
        log.info(
            "[%3d] %s r%-2s %-30s %-7s laps=%-5s drv=%-3s %5.1fs %s",
            n, rec.year, rec.round, rec.event_name[:30], rec.status,
            rec.n_laps, rec.n_drivers, time.time() - t0, rec.error[:70],
        )
        records.append(asdict(rec))
        # Write after every session so a crash never loses the manifest.
        pd.DataFrame(records).to_csv(manifest_path, index=False)
        time.sleep(SLEEP_BETWEEN_SESSIONS_S)

    df = pd.DataFrame(records)
    if not df.empty:
        log.info("status counts:\n%s", df["status"].value_counts().to_string())
        log.info("manifest -> %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
