"""Validation V6 - does the simulator price strategy the way real races do?

Why this exists
---------------
Every component model is validated on its own terms: the degradation slope
predicts held-out lap times with a calibration coefficient of 1.02, pit loss
and caution hazard come straight from the timing data. None of that says the
*simulator* turns those parts into the right race-level cost of a strategy,
and the strategy layer is where decisions - and the sanity gate - live.

So this compares two strategy-level quantities, measured the same way in real
races and in simulated replays of the same races, on the same cars:

  extra stop       seconds of finishing time per additional pit stop
  stint balance    seconds of finishing time per unit of longest-stint share,
                   among cars that made the same number of stops

Both are within-race comparisons (race fixed effects, controlling for grid
position), with cars whose extra stop was forced rather than chosen removed.
Intervals are cluster bootstraps over races.

What it found
-------------
Real races put both near zero: an extra stop costs nothing measurable, and
how stints are split costs nothing measurable, across the range teams run.
The simulator charges ~10 s per extra stop and ~100+ s per unit of stint
share. Calibrating one knob to match the first (a stint-length curvature, or
a lower effective pit loss) leaves the second far outside its interval - the
curvature made it worse - so neither was adopted.

The reading: per-lap degradation is estimated correctly, but the simulator
assumes every car runs flat out on the fitted wear rate for the whole stint.
Real drivers manage tyres, and a long stint happens because the tyre is
lasting. That is a structural limit of the simulator, not a parameter, and it
is why the counterfactual audit is withheld.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("pitwall.models.strategy_validation")

# A stint of five laps or fewer, or a stop inside the last five laps, is a
# puncture, damage, a penalty or a late free stop - not a strategy choice.
UNPLANNED_MIN_STINT = 5
UNPLANNED_LAST_STOP_FROM_END = 5

MIN_FINISHED_SHARE = 0.95
MAX_STOPS_CONSIDERED = 3


def car_race_table(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """One row per classified, planned car-race: real finishing time, stops, grid.

    Retirements are dropped (a car that stopped on lap 30 did not run a
    strategy) and so are the forced stops described above.
    """
    d = laps.assign(lap_s=pd.to_numeric(laps["LapTime"], errors="coerce"))
    race_laps = d.groupby("race_id")["race_laps"].median()

    car = (
        d.groupby(["race_id", "circuit", "car_id"])
        .agg(
            real_time_s=("lap_s", "sum"),
            n_laps=("LapNumber", "max"),
            n_valid=("lap_s", lambda s: s.notna().sum()),
        )
        .reset_index()
    )
    car["race_laps"] = car["race_id"].map(race_laps)
    car = car[
        (car["n_laps"] >= MIN_FINISHED_SHARE * car["race_laps"])
        & (car["n_valid"] >= car["n_laps"] - 2)
    ]

    s = stints.dropna(subset=["stint_length"])
    agg = (
        s.groupby(["race_id", "car_id"])
        .agg(n_stops=("ended_in_pit", "sum"), min_stint=("stint_length", "min"))
        .reset_index()
    )
    last_stop = (
        s[s["ended_in_pit"].astype(bool)]
        .groupby(["race_id", "car_id"])["end_lap"]
        .max()
        .rename("last_stop")
    )
    agg = agg.merge(last_stop, on=["race_id", "car_id"], how="left")
    agg["race_laps"] = agg["race_id"].map(race_laps)
    agg["unplanned"] = (agg["min_stint"] <= UNPLANNED_MIN_STINT) | (
        (agg["race_laps"] - agg["last_stop"]) <= UNPLANNED_LAST_STOP_FROM_END
    ).fillna(False)

    grid = d[d["LapNumber"] == 1].groupby(["race_id", "car_id"])["Position"].first().rename("grid")

    out = car.merge(
        agg[["race_id", "car_id", "n_stops", "unplanned"]], on=["race_id", "car_id"]
    ).merge(grid, on=["race_id", "car_id"], how="left")
    return out[~out["unplanned"] & out["n_stops"].between(1, MAX_STOPS_CONSIDERED)].dropna(
        subset=["grid", "real_time_s"]
    )


def _blocks(t: pd.DataFrame, time_col: str, x_col: str, within: list[str]) -> list[np.ndarray]:
    """Demeaned (gap, x, grid) per fixed-effect group, ready to stack.

    Demeaning inside each group is the fixed effect; keeping groups as blocks
    lets the bootstrap resample whole races cheaply. Groups with no variation
    in ``x`` carry no information and are dropped.
    """
    out = []
    for _, g in t.groupby(within, sort=False):
        if len(g) < 3 or g[x_col].nunique() < 2:
            continue
        m = g[[time_col, x_col, "grid"]].to_numpy(float)
        if not np.isfinite(m).all():
            continue
        m[:, 0] -= m[:, 0].min()
        out.append(m - m.mean(axis=0))
    return out


def _slope(blocks: list[np.ndarray]) -> float:
    if len(blocks) < 3:
        return float("nan")
    m = np.vstack(blocks)
    return float(np.linalg.lstsq(m[:, 1:], m[:, 0], rcond=None)[0][0])


def _estimate(
    t: pd.DataFrame, time_col: str, x_col: str, within: list[str], n_boot: int, seed: int
) -> dict:
    """Point estimate and a cluster bootstrap over races."""
    by_race: dict[str, list[np.ndarray]] = {}
    for rid, g in t.groupby("race_id", sort=False):
        b = _blocks(g, time_col, x_col, within)
        if b:
            by_race[rid] = b
    races = list(by_race)
    point = _slope([b for r in races for b in by_race[r]])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(races), len(races)) if races else []
        v = _slope([b for i in pick for b in by_race[races[i]]])
        if np.isfinite(v):
            boot.append(v)
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return {"estimate": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_races": len(races)}


def _difference(t: pd.DataFrame, x_col: str, within: list[str], n_boot: int, seed: int) -> dict:
    """Simulated minus real, with races resampled jointly for both sides.

    Two separately bootstrapped intervals can overlap while the difference is
    still clearly non-zero, because both sides share the same races. Pairing
    the resample is the correct test.
    """
    by_race = {}
    for rid, g in t.groupby("race_id", sort=False):
        s, r = _blocks(g, "sim_time_s", x_col, within), _blocks(g, "real_time_s", x_col, within)
        if s and r:
            by_race[rid] = (s, r)
    races = list(by_race)

    def diff(ids):
        return _slope([b for i in ids for b in by_race[races[i]][0]]) - _slope(
            [b for i in ids for b in by_race[races[i]][1]]
        )

    point = diff(range(len(races)))
    rng = np.random.default_rng(seed)
    boot = [diff(rng.integers(0, len(races), len(races))) for _ in range(n_boot)]
    boot = [b for b in boot if np.isfinite(b)]
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return {"estimate": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_races": len(races)}


def moments(t: pd.DataFrame, time_col: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """Both strategy-level moments for one set of finishing times."""
    return {
        "extra_stop_s": _estimate(t, time_col, "n_stops", ["race_id"], n_boot, seed),
        "stint_balance_s": _estimate(
            t, time_col, "longest_stint_share", ["race_id", "n_stops"], n_boot, seed
        ),
    }


def real_vs_simulated(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    artefacts: dict,
    race_ids: list[str],
    n_sims: int = 200,
    seed: int = 7,
    n_boot: int = 1000,
) -> dict:
    """Replay each race with every car on its real strategy; compare the moments."""
    from pitwall import audit, pipeline
    from pitwall.sim import simulate

    real = car_race_table(laps[laps["race_id"].isin(race_ids)], stints)
    circuit_of = laps.groupby("race_id")["circuit"].first()

    rows = []
    for rid in race_ids:
        st = audit.reconstruct_strategies(laps, rid)
        if st.empty:
            continue
        race_laps = int(st["race_laps"].iloc[0])
        field = audit.build_field(laps, rid, st)
        if len(field) < 6:
            continue
        params = pipeline.circuit_params(circuit_of[rid], artefacts)
        if race_laps >= 10:
            params.race_laps = race_laps
        finish = simulate(params, field, n_sims=n_sims, seed=seed).finish_times.mean(axis=0)
        for i, r in st.reset_index(drop=True).iterrows():
            edges = [0, *(lap for lap, _ in r["strategy"].stops), race_laps]
            longest = max(edges[k + 1] - edges[k] for k in range(len(edges) - 1))
            rows.append(
                {
                    "race_id": rid,
                    "car_id": r["car_id"],
                    "reconstructed_stops": len(r["strategy"].stops),
                    "longest_stint_share": longest / race_laps,
                    "sim_time_s": float(finish[i]),
                }
            )

    t = pd.DataFrame(rows).merge(real, on=["race_id", "car_id"])
    # Keep only cars whose reconstructed plan agrees with the stint table, so
    # the real and simulated sides are describing the same strategy.
    t = t[t["reconstructed_stops"] == t["n_stops"]]
    log.info("strategy validation: %d car-races in %d races", len(t), t["race_id"].nunique())

    return {
        "n_car_races": len(t),
        "n_races": int(t["race_id"].nunique()),
        "real": moments(t, "real_time_s", n_boot=n_boot),
        "simulated": moments(t, "sim_time_s", n_boot=n_boot),
        "simulated_minus_real": {
            "extra_stop_s": _difference(t, "n_stops", ["race_id"], n_boot, 0),
            "stint_balance_s": _difference(
                t, "longest_stint_share", ["race_id", "n_stops"], n_boot, 0
            ),
        },
    }
