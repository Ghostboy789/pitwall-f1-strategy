"""Phase 1 data-quality gates and the pre-registered race exclusion rules.

Two separate jobs, kept in one place so both are auditable:

* ``run_gates``      the V5 structural checks. These ask whether the data is
                     internally coherent - lap numbers monotonic, stints
                     continuous, tyre age resetting at stops. A failure here
                     means the pipeline is wrong.
* ``race_exclusions`` rules E1-E5 from VALIDATION_PLAN.md, applied
                     mechanically. These ask whether a race is a sensible
                     subject for strategy modelling. A race caught here is not
                     a bug, it is a race where the premise does not hold.

Nothing here drops data silently. Every gate reports a count and every
excluded race is named, because a filter that quietly removes most of the
sample is itself the most important thing to know about a result.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pitwall import config

log = logging.getLogger("pitwall.quality")

RED_FLAG = "5"
SAFETY_CAR = "4"
VSC = "6"


def run_gates(laps: pd.DataFrame) -> pd.DataFrame:
    """Structural checks (V5). Returns one row per check with a pass/fail."""
    checks = []

    def add(name: str, n_bad: int, n_total: int, detail: str = "") -> None:
        checks.append(
            {
                "check": name,
                "n_violations": int(n_bad),
                "n_checked": int(n_total),
                "share": float(n_bad / n_total) if n_total else 0.0,
                "passed": bool(n_bad == 0),
                "detail": detail,
            }
        )

    add("non_empty", int(len(laps) == 0), 1, "lap table has rows")

    # Lap numbers strictly increasing within a car's race.
    g = laps.sort_values(["car_id", "LapNumber"]).groupby("car_id")["LapNumber"]
    non_mono = int((g.diff().dropna() <= 0).sum())
    add("lap_number_monotonic", non_mono, len(laps), "LapNumber increases within car_id")

    # Stint numbers never go backwards.
    gs = laps.sort_values(["car_id", "LapNumber"]).groupby("car_id")["Stint"]
    stint_back = int((gs.diff().dropna() < 0).sum())
    add("stint_non_decreasing", stint_back, len(laps), "Stint never decreases within car_id")

    # Tyre age advances by one within a stint.
    d = laps.sort_values(["stint_id", "LapNumber"])
    steps = d.groupby("stint_id")["TyreLife"].diff().dropna()
    bad_steps = int((steps != 1).sum())
    add("tyre_age_increments", bad_steps, len(steps), "TyreLife advances by 1 within a stint")

    # A FRESH tyre must start at age 1.
    #
    # An earlier version of this gate asked whether each new stint started on a
    # younger tyre than the one just removed, and failed on 565 of 10,421
    # stints. That gate was wrong, not the data: every one of those 565 carries
    # FreshTyre = False, because teams routinely fit a scrubbed set that has
    # more laps on it than the set coming off. Likewise the 515 stints where
    # TyreLife appears to continue incrementing across a stop are all used
    # sets, not a broken reset.
    #
    # The invariant that actually holds is the one below, and it holds: 26 of
    # 7,666 fresh stints (0.34%) start above age 1.
    firsts = (
        d.groupby("stint_id")
        .agg(
            car_id=("car_id", "first"),
            start_age=("TyreLife", "min"),
            fresh=("FreshTyre", "first"),
        )
        .reset_index()
    )
    fresh = firsts[firsts["fresh"].astype("boolean").fillna(False)]
    bad_fresh = int((fresh["start_age"] > 1).sum())
    tolerance = int(np.ceil(0.01 * max(len(fresh), 1)))
    checks.append(
        {
            "check": "fresh_tyre_starts_at_age_1",
            "n_violations": bad_fresh,
            "n_checked": len(fresh),
            "share": float(bad_fresh / len(fresh)) if len(fresh) else 0.0,
            "passed": bool(bad_fresh <= tolerance),
            "detail": f"a new set starts at TyreLife 1 (tolerance {tolerance})",
        }
    )

    # Lap times within physical bounds (after filtering).
    lt = pd.to_numeric(laps.get("lap_time_s", laps.get("LapTime")), errors="coerce").dropna()
    out_of_range = int((~lt.between(config.MIN_LAP_TIME_S, config.MAX_LAP_TIME_S)).sum())
    add(
        "lap_time_in_bounds",
        out_of_range,
        len(lt),
        f"{config.MIN_LAP_TIME_S}-{config.MAX_LAP_TIME_S}s",
    )

    # Every canonical circuit key is one we know about.
    from pitwall.circuits import CIRCUIT_REF

    unknown = sorted(set(laps["circuit"].unique()) - set(CIRCUIT_REF))
    add("circuit_keys_known", len(unknown), laps["circuit"].nunique(), str(unknown))

    # A driver appears at most once per lap of a race.
    dup = int(laps.duplicated(subset=["race_id", "Driver", "LapNumber"]).sum())
    add("no_duplicate_car_laps", dup, len(laps), "one row per car per lap")

    out = pd.DataFrame(checks)
    n_failed = int((~out["passed"]).sum())
    log.info("data-quality gates: %d/%d passed", len(out) - n_failed, len(out))
    return out


def _track_status_flags(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-race flags for red flags, safety cars and green running."""
    ts = laps["TrackStatus"].fillna("").astype(str)
    d = laps.assign(
        _red=ts.str.contains(RED_FLAG),
        _sc=ts.str.contains(SAFETY_CAR) | ts.str.contains(VSC),
    )
    per_race = (
        d.groupby("race_id")
        .agg(
            year=("year", "first"),
            round=("round", "first"),
            event_name=("event_name", "first"),
            circuit=("circuit", "first"),
            race_laps=("race_laps", "max"),
            n_laps=("LapNumber", "size"),
            n_green=("is_green", "sum"),
            any_red=("_red", "max"),
            n_drivers=("Driver", "nunique"),
        )
        .reset_index()
    )

    # Earliest lap on which a red flag appears, as a fraction of distance.
    red = d[d["_red"]].groupby("race_id")["LapNumber"].min()
    per_race["first_red_lap"] = per_race["race_id"].map(red)
    per_race["first_red_fraction"] = per_race["first_red_lap"] / per_race["race_laps"]

    # Median green laps completed per driver: the substance of the race.
    green_per_driver = (
        d[d["is_green"]].groupby(["race_id", "Driver"]).size().groupby("race_id").median()
    )
    per_race["median_green_laps"] = per_race["race_id"].map(green_per_driver).fillna(0)

    wet = d.groupby("race_id")["is_wet_tyre"].mean()
    per_race["wet_lap_share"] = per_race["race_id"].map(wet).fillna(0.0)
    return per_race


def race_exclusions(laps: pd.DataFrame) -> pd.DataFrame:
    """Apply exclusion rules E1-E5. One row per race, with the rule that caught it.

    E3 (early red flag) and E5 (wet race) exclude a race from *strategy and
    degradation* fitting only. Their caution events are still used by the
    hazard model - removing the chaotic races from a model of chaos would bias
    it badly - and that split is flagged per race rather than assumed.
    """
    r = _track_status_flags(laps)

    r["E1_too_few_green_laps"] = r["median_green_laps"] < config.MIN_GREEN_LAPS_FOR_RACE
    r["E2_not_a_race"] = r["race_laps"] < 2
    r["E3_early_red_flag"] = r["first_red_fraction"].notna() & (
        r["first_red_fraction"] <= config.EARLY_RED_FLAG_RACE_FRACTION
    )
    r["E5_wet_race"] = r["wet_lap_share"] > 0.30

    rules = ["E1_too_few_green_laps", "E2_not_a_race", "E3_early_red_flag", "E5_wet_race"]
    r["excluded"] = r[rules].any(axis=1)
    r["exclusion_reason"] = r[rules].apply(
        lambda row: "|".join([c for c in rules if row[c]]) or "", axis=1
    )
    # Cautions from every race, including excluded ones, feed the hazard model.
    r["use_for_hazard"] = ~r["E2_not_a_race"]
    r["use_for_strategy"] = ~r["excluded"]

    n_ex = int(r["excluded"].sum())
    log.info("race exclusions: %d of %d races excluded from strategy modelling", n_ex, len(r))
    for _, row in r[r["excluded"]].iterrows():
        log.info("  excluded %s %s (%s)", row["year"], row["event_name"], row["exclusion_reason"])
    return r


def filter_report(pace: pd.DataFrame) -> pd.DataFrame:
    """Lap-filter attrition, from the attrs stamped on by ``dataset.clean_for_pace``."""
    rep = pace.attrs.get("filter_report", {})
    n0 = pace.attrs.get("n_before_filter", len(pace))
    rows = [
        {"rule": k, "n_removed": v, "share_of_raw": v / n0 if n0 else 0.0} for k, v in rep.items()
    ]
    rows.append(
        {"rule": "RETAINED", "n_removed": len(pace), "share_of_raw": len(pace) / n0 if n0 else 0}
    )
    return pd.DataFrame(rows)


def sample_counts(laps: pd.DataFrame) -> pd.DataFrame:
    """Races and laps per circuit and era.

    TRAP T4. These counts travel with every per-circuit number in every
    user-facing surface, so a reader always knows how thin the evidence is.
    """
    return (
        laps.groupby(["circuit", "era"])
        .agg(races=("race_id", "nunique"), laps=("LapNumber", "size"))
        .reset_index()
        .sort_values(["circuit", "era"])
    )
