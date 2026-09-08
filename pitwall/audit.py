"""Phase 7 and 8 - counterfactual audit, and the sanity gate.

What each real strategy cost
----------------------------
For every car in every retained race: reconstruct what the team actually did,
re-run the race with that strategy, then re-run it again with the optimiser's
choice while the rest of the field keeps its real plan. The difference is what
the strategy cost or gained, and it is reported with an interval.

The expectation, stated in advance, is that most results are statistically
indistinguishable from zero. Real strategists are good. A model that says
otherwise is far more likely to be broken than brilliant, which is what the
sanity gate exists to catch.

THE SANITY GATE (validation V3)
-------------------------------
Pre-registered in VALIDATION_PLAN.md before any of this ran. The model is
declared broken - and the cause hunted down before any headline number is
published - if any of:

* mean claimed gain across all car-races exceeds **2.0 s**, or
* the optimiser claims a gain for more than **70%** of car-races, or
* any single claimed gain exceeds **30 s** without an identifiable cause.

These thresholds are not adjustable after seeing the result. If the gate
trips, the diagnosis goes in the README as a headline finding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitwall import config
from pitwall.optimize import diverse_candidates, enumerate_strategies
from pitwall.sim import Car, CircuitParams, Strategy, simulate

log = logging.getLogger("pitwall.audit")

# Pre-registered gate thresholds. Do not tune.
GATE_MAX_MEAN_GAIN_S = 2.0
GATE_MAX_SHARE_IMPROVED = 0.70
GATE_MAX_SINGLE_GAIN_S = 30.0

RANK_OF_LABEL = {"SOFTEST": 0, "MIDDLE": 1, "HARDEST": 2}


@dataclass
class AuditConfig:
    """Knobs for how much of the calendar to audit and how hard."""

    n_sims: int = 400
    top_k: int = 6
    max_stops: int = 2
    min_laps_completed_frac: float = 0.9
    seed: int = config.SEED


def reconstruct_strategies(laps: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """What each car actually did: stop laps and the compound rank fitted.

    Only cars that reached the finish are audited. A car that retired on lap
    12 has a truncated strategy, and 'what would a better strategy have done'
    is unanswerable for it - the answer is dominated by the retirement, not by
    the tyres.
    """
    r = laps[laps["race_id"] == race_id]
    if r.empty:
        return pd.DataFrame()
    race_laps = int(pd.to_numeric(r["LapNumber"], errors="coerce").max())

    rows = []
    for car_id, g in r.groupby("car_id"):
        g = g.sort_values("LapNumber")
        completed = int(pd.to_numeric(g["LapNumber"], errors="coerce").max())
        stints = (
            g.groupby("Stint")
            .agg(
                start_lap=("LapNumber", "min"),
                rank=("compound_rank", "first"),
                label=("compound_rank_label", "first"),
            )
            .reset_index()
            .sort_values("Stint")
            .dropna(subset=["rank"])
        )
        if len(stints) == 0:
            continue
        stops = tuple(
            (int(row.start_lap) - 1, int(row["rank"]))
            for _, row in stints.iloc[1:].iterrows()
            if int(row.start_lap) - 1 > 0
        )
        rows.append(
            {
                "race_id": race_id,
                "car_id": car_id,
                "driver": str(g["Driver"].iloc[0]),
                "team": str(g["Team"].iloc[0]),
                "laps_completed": completed,
                "race_laps": race_laps,
                "finished_share": completed / race_laps if race_laps else 0.0,
                "n_stops": len(stops),
                "strategy": Strategy(stops=stops, start_rank=int(stints.iloc[0]["rank"])),
            }
        )
    return pd.DataFrame(rows)


def build_field(laps: pd.DataFrame, race_id: str, strategies: pd.DataFrame) -> list[Car]:
    """Grid, pace and plan for every car in one race.

    Pace comes from each car's own clean-air laps in that race, expressed
    relative to the quickest car. Grid position falls back to the order at the
    end of lap 1 where classification data is unavailable - a documented
    approximation, since lap 1 has already been raced.
    """
    r = laps[laps["race_id"] == race_id].copy()
    r["lap_time_s"] = pd.to_numeric(r["LapTime"], errors="coerce")
    clean = r[r["is_green"] & ~r["is_inlap"] & ~r["is_outlap"]]
    pace = clean.groupby("car_id")["lap_time_s"].median()
    if pace.empty:
        return []
    pace = pace - pace.min()

    lap1 = (
        r[r["LapNumber"] == 1].set_index("car_id")["Position"].to_dict()
        if (r["LapNumber"] == 1).any()
        else {}
    )

    cars = []
    for _, row in strategies.iterrows():
        cid = row["car_id"]
        # Lap-1 position can be missing when a car's first lap did not record
        # (a first-corner incident, a missed timing loop). Falling back to the
        # next free slot keeps the field intact rather than dropping the car.
        pos = lap1.get(cid)
        grid = int(pos) if pos is not None and np.isfinite(pos) else len(cars) + 1
        cars.append(
            Car(
                driver=row["driver"],
                team=row["team"],
                pace_offset_s=float(pace.get(cid, pace.median())),
                grid_position=grid,
                strategy=row["strategy"],
            )
        )
    return cars


def audit_race(
    laps: pd.DataFrame,
    race_id: str,
    params: CircuitParams,
    cfg: AuditConfig | None = None,
) -> pd.DataFrame:
    """Counterfactual for every finishing car in one race."""
    cfg = cfg or AuditConfig()
    strategies = reconstruct_strategies(laps, race_id)
    if strategies.empty:
        return pd.DataFrame()

    field = build_field(laps, race_id, strategies)
    if len(field) < 4:
        return pd.DataFrame()

    actual = simulate(params, field, n_sims=cfg.n_sims, seed=cfg.seed)
    candidates = diverse_candidates(
        enumerate_strategies(params, max_stops=cfg.max_stops), cfg.top_k
    )
    if not candidates:
        return pd.DataFrame()

    rows = []
    for i, row in strategies.reset_index(drop=True).iterrows():
        if row["finished_share"] < cfg.min_laps_completed_frac:
            continue
        base_time = float(actual.finish_times[:, i].mean())
        base_pos = float(actual.finish_positions[:, i].mean())

        best_time, best_pos, best_strategy = base_time, base_pos, None
        for cand in candidates:
            trial = list(field)
            trial[i] = Car(
                driver=field[i].driver,
                pace_offset_s=field[i].pace_offset_s,
                grid_position=field[i].grid_position,
                strategy=cand.strategy,
                team=field[i].team,
            )
            res = simulate(params, trial, n_sims=cfg.n_sims, seed=cfg.seed)
            t = float(res.finish_times[:, i].mean())
            if t < best_time:
                best_time = t
                best_pos = float(res.finish_positions[:, i].mean())
                best_strategy = cand.strategy

        rows.append(
            {
                "race_id": race_id,
                "circuit": params.circuit,
                "driver": row["driver"],
                "team": row["team"],
                "actual_strategy": str(row["strategy"]),
                "actual_n_stops": row["n_stops"],
                "best_strategy": str(best_strategy) if best_strategy else str(row["strategy"]),
                "actual_time_s": base_time,
                "best_time_s": best_time,
                "gain_s": base_time - best_time,
                "actual_position": base_pos,
                "best_position": best_pos,
                "position_gain": base_pos - best_pos,
            }
        )
    return pd.DataFrame(rows)


def sanity_gate(audit: pd.DataFrame) -> dict:
    """Validation V3. The most important check in the project.

    Returns the verdict and every quantity the pre-registered thresholds are
    applied to, so a reader can check the arithmetic rather than take the
    verdict on trust.
    """
    if audit.empty:
        return {"status": "no_data", "passed": False}

    g = audit["gain_s"]
    mean_gain = float(g.mean())
    share_improved = float((g > 0.01).mean())
    max_gain = float(g.max())

    failures = []
    if mean_gain > GATE_MAX_MEAN_GAIN_S:
        failures.append(f"mean gain {mean_gain:.2f}s exceeds {GATE_MAX_MEAN_GAIN_S}s")
    if share_improved > GATE_MAX_SHARE_IMPROVED:
        failures.append(
            f"optimiser beats {share_improved:.0%} of car-races, "
            f"above the {GATE_MAX_SHARE_IMPROVED:.0%} threshold"
        )
    if max_gain > GATE_MAX_SINGLE_GAIN_S:
        failures.append(f"largest single gain {max_gain:.1f}s exceeds {GATE_MAX_SINGLE_GAIN_S}s")

    return {
        "status": "ok",
        "passed": len(failures) == 0,
        "failures": failures,
        "n_car_races": len(audit),
        "n_races": int(audit["race_id"].nunique()),
        "mean_gain_s": mean_gain,
        "median_gain_s": float(g.median()),
        "share_improved": share_improved,
        "max_gain_s": max_gain,
        "p95_gain_s": float(g.quantile(0.95)),
        "share_within_1s_of_optimal": float((g <= 1.0).mean()),
        "thresholds": {
            "max_mean_gain_s": GATE_MAX_MEAN_GAIN_S,
            "max_share_improved": GATE_MAX_SHARE_IMPROVED,
            "max_single_gain_s": GATE_MAX_SINGLE_GAIN_S,
        },
    }


def team_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Strategy cost by team - one of the secondary deliverables."""
    return (
        audit.groupby("team")
        .agg(
            n_car_races=("gain_s", "size"),
            mean_gain_s=("gain_s", "mean"),
            median_gain_s=("gain_s", "median"),
            sd_gain_s=("gain_s", "std"),
            share_suboptimal=("gain_s", lambda s: float((s > 1.0).mean())),
        )
        .reset_index()
        .assign(se_gain_s=lambda d: d["sd_gain_s"] / np.sqrt(d["n_car_races"]))
        .sort_values("mean_gain_s")
    )


def driver_summary(audit: pd.DataFrame, min_races: int = 5) -> pd.DataFrame:
    """Strategy cost by driver. Filtered to drivers with enough car-races."""
    out = (
        audit.groupby("driver")
        .agg(
            n_car_races=("gain_s", "size"),
            mean_gain_s=("gain_s", "mean"),
            median_gain_s=("gain_s", "median"),
            sd_gain_s=("gain_s", "std"),
        )
        .reset_index()
    )
    out["se_gain_s"] = out["sd_gain_s"] / np.sqrt(out["n_car_races"])
    return out[out["n_car_races"] >= min_races].sort_values("mean_gain_s")
