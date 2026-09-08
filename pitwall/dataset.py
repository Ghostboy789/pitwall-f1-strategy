"""Assemble the analysis table from ingested parquet.

This module owns every transformation between raw FastF1 output and the
lap-level table the models consume, so the filters are applied in exactly one
place and can be audited in exactly one place.

The filters implement the lap-level rules L1-L6 pre-registered in
VALIDATION_PLAN.md. Nothing here is tuned; the thresholds live in
``pitwall.config`` and were fixed before any result was computed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pitwall import config
from pitwall.circuits import canonical_circuit

log = logging.getLogger("pitwall.dataset")

# FastF1 emits TrackStatus as a string of concatenated status codes for the
# lap, e.g. "1" (all clear) or "1245" (multiple states during that lap).
# A lap is only green if nothing but "1" appears.
GREEN = "1"


def load_raw(kind: str = "laps") -> pd.DataFrame:
    """Concatenate every ingested parquet of one kind."""
    d = config.RAW / kind
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {kind} parquet in {d}. Run `python -m pitwall.ingest` first.")
    frames = [pd.read_parquet(f) for f in files]
    out = pd.concat(frames, ignore_index=True)
    log.info("loaded %s: %d rows from %d sessions", kind, len(out), len(files))
    return out


def add_identity(laps: pd.DataFrame) -> pd.DataFrame:
    """Add canonical circuit key and a stable per-race / per-car identifier."""
    out = laps.copy()
    out["circuit"] = [
        canonical_circuit(loc, evt) for loc, evt in zip(out["location"], out["event_name"])
    ]
    out["race_id"] = out["year"].astype(str) + "_" + out["round"].astype(str).str.zfill(2)
    out["car_id"] = out["race_id"] + "_" + out["Driver"].astype(str)
    out["stint_id"] = out["car_id"] + "_s" + out["Stint"].astype("Int64").astype(str)
    return out


def is_green_lap(track_status: pd.Series) -> pd.Series:
    """True where the lap ran entirely under green flag.

    TrackStatus is a concatenation of every status code seen during the lap,
    so a lap is green only if the string contains nothing but '1'.
    """
    s = track_status.fillna("").astype(str)
    return s.str.replace(GREEN, "", regex=False).eq("") & s.ne("")


def add_race_features(laps: pd.DataFrame) -> pd.DataFrame:
    """Add per-race progression, fuel proxy and pit-cycle flags.

    Fuel note (TRAP T5). There is no refuelling in this era, so fuel mass is a
    deterministic function of laps completed and cannot be observed directly.
    ``fuel_laps_burned`` is the physical proxy: laps completed so far.

    Fuel burn and track evolution are both monotone in lap number and are
    therefore *not separately identified within a single race*. They are
    separated from tyre degradation because stint timing varies across
    drivers - at any given lap number the field carries a wide spread of tyre
    ages - which is what makes the decomposition identifiable at all. The
    split between fuel and evolution *within* the lap-number term is handled,
    and its limits stated, in ``pitwall.models.pace``.
    """
    out = laps.copy()

    total = out.groupby("race_id")["LapNumber"].transform("max")
    out["race_laps"] = total
    out["race_progress"] = out["LapNumber"] / total
    out["fuel_laps_burned"] = out["LapNumber"] - 1
    out["fuel_laps_remaining"] = total - out["LapNumber"]

    out["is_green"] = is_green_lap(out["TrackStatus"])
    out["is_inlap"] = out["PitInTime"].notna()
    out["is_outlap"] = out["PitOutTime"].notna()
    out["is_first_lap"] = out["LapNumber"] <= 1

    comp = out["Compound"].astype(str).str.upper()
    out["Compound"] = comp
    out["is_wet_tyre"] = comp.isin(config.WET_LABELS)
    out["valid_compound"] = (
        ~comp.isin([c.upper() for c in config.INVALID_COMPOUNDS if c])
        & comp.ne("NAN")
        & comp.ne("NONE")
    )

    return out


def add_traffic(laps: pd.DataFrame) -> pd.DataFrame:
    """Add a dirty-air proxy: time gap to the car immediately ahead.

    Within a race, cars are ordered by the session time at which they crossed
    the line on each lap number; the gap to the car ahead in that ordering is
    the dirty-air exposure for the following lap. This is a proxy, not a
    measurement: it uses the gap at the line rather than the average gap
    through the lap, and it does not know whether the car ahead was actually
    on the racing line. Its limitations are stated in the methodology report.

    ``np.inf`` for the leader, who has clear air by definition.
    """
    out = laps.sort_values(["race_id", "LapNumber", "Time"]).copy()
    grp = out.groupby(["race_id", "LapNumber"], sort=False)
    out["gap_ahead_s"] = out["Time"] - grp["Time"].shift(1)
    out["track_position"] = grp.cumcount() + 1
    out.loc[out["track_position"] == 1, "gap_ahead_s"] = np.inf

    # Dirty air is generally held to matter within roughly 1.5-2.0 s. The
    # continuous gap is kept as well so the model can estimate the shape
    # rather than inherit this cut-off as an assumption.
    out["in_dirty_air"] = out["gap_ahead_s"] < 2.0
    out["dirty_air_intensity"] = np.where(
        np.isfinite(out["gap_ahead_s"]),
        np.exp(-out["gap_ahead_s"].clip(lower=0) / 1.5),
        0.0,
    )
    return out


def clean_for_pace(laps: pd.DataFrame) -> pd.DataFrame:
    """Apply pre-registered lap filters L1-L5 for pace modelling.

    Returns the retained laps. The count removed by each rule is logged, and
    surfaced by ``pitwall.quality.filter_report`` for the methodology report -
    filters that silently discard most of the data are a failure mode, so the
    numbers are published.
    """
    out = laps.copy()
    n0 = len(out)

    lt = pd.to_numeric(out["LapTime"], errors="coerce")
    out["lap_time_s"] = lt

    masks = {
        "L1_implausible_laptime": lt.between(config.MIN_LAP_TIME_S, config.MAX_LAP_TIME_S),
        "L2_non_green": out["is_green"],
        "L3_pit_cycle": ~(out["is_inlap"] | out["is_outlap"]),
        "L4_first_lap": ~out["is_first_lap"],
        "L5_invalid_compound": out["valid_compound"] & ~out["is_wet_tyre"],
    }

    keep = pd.Series(True, index=out.index)
    report = {}
    for name, m in masks.items():
        m = m.fillna(False)
        report[name] = int((keep & ~m).sum())
        keep &= m

    out = out[keep].copy()
    log.info(
        "pace filter: %d -> %d laps (%.1f%% retained); removed by rule: %s",
        n0,
        len(out),
        100 * len(out) / max(n0, 1),
        report,
    )
    out.attrs["filter_report"] = report
    out.attrs["n_before_filter"] = n0
    return out


def stint_table(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per stint, for the survival / censoring model (TRAP T2).

    A stint is *censored* when it ended for a reason other than a strategic
    tyre change - the chequered flag, a retirement, or a red flag. Those
    stints tell us the tyre lasted at least that long, not that the team
    judged it finished. Treating them as ordinary observations is exactly the
    bias the survival model exists to remove.
    """
    g = laps.sort_values(["stint_id", "LapNumber"]).groupby("stint_id", sort=False)

    out = pd.DataFrame(
        {
            "race_id": g["race_id"].first(),
            "car_id": g["car_id"].first(),
            "year": g["year"].first(),
            "round": g["round"].first(),
            "circuit": g["circuit"].first(),
            "era": g["era"].first(),
            "driver": g["Driver"].first(),
            "team": g["Team"].first(),
            "stint": g["Stint"].first(),
            "compound": g["Compound"].first(),
            "fresh_tyre": g["FreshTyre"].first(),
            "start_lap": g["LapNumber"].min(),
            "end_lap": g["LapNumber"].max(),
            "start_tyre_life": g["TyreLife"].min(),
            "end_tyre_life": g["TyreLife"].max(),
            "n_laps": g["LapNumber"].size(),
            "race_laps": g["race_laps"].first(),
            "ended_in_pit": g["is_inlap"].max(),
            "any_non_green": (~g["is_green"].min().astype(bool)),
        }
    ).reset_index()

    out["stint_length"] = out["end_tyre_life"] - out["start_tyre_life"] + 1

    # A stint is an uncensored observation of the pit decision only if it
    # actually ended in a pit stop. Everything else is right-censored.
    out["event_observed"] = out["ended_in_pit"].astype(bool)
    out["reached_race_end"] = out["end_lap"] >= out["race_laps"] - 1
    out.loc[out["reached_race_end"], "event_observed"] = False

    return out


def build(save: bool = True) -> dict[str, pd.DataFrame]:
    """Build every analysis table from raw parquet. One command, per the brief."""
    laps = load_raw("laps")
    laps = add_identity(laps)
    laps = add_race_features(laps)
    laps = add_traffic(laps)

    pace = clean_for_pace(laps)
    stints = stint_table(laps)

    if save:
        config.PROCESSED.mkdir(parents=True, exist_ok=True)
        laps.to_parquet(config.PROCESSED / "laps_all.parquet", index=False)
        pace.to_parquet(config.PROCESSED / "laps_pace.parquet", index=False)
        stints.to_parquet(config.PROCESSED / "stints.parquet", index=False)
        log.info("wrote analysis tables to %s", config.PROCESSED)

    return {"laps": laps, "pace": pace, "stints": stints}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    tables = build()
    for name, t in tables.items():
        print(f"{name:8s} {t.shape}")
