"""A stint-length cost term, calibrated to what pit stops cost in real races.

The problem this exists to solve
--------------------------------
The linear simulator priced an extra pit stop at +10.3 s of finishing time.
Real races, measured the same way on the same 2018-23 races, put it at +1.1 s
(95% CI -5.4 to +7.8), and a team-mate comparison - same car, same race,
different stop count - at -1.2 s. That mis-pricing is the sanity-gate
failure: the optimiser preferred one fewer stop than teams made, and each
missing stop surfaced as roughly one pit loss of phantom gain.

It was not the degradation slope. The slope predicts held-out lap times at
tyre ages 3-15 with a coefficient of 1.02 (target 1.0), per-circuit slope
error does not correlate with claimed gain (r = -0.02), and the gain survives
re-scoring on a fresh random seed almost untouched (0.33 s of 16 s).

What this module does
---------------------
``observed_cost_of_extra_stop`` measures the real cost of a stop from
finishing times, with race fixed effects and forced stops removed. The same
regression on *simulated* finishing times gives the model's answer, and one
global coefficient on (tyre age)^2 is solved for so the two agree. This is
indirect inference: a parameter pinned by a race-level moment rather than
fitted to lap times.

What it is NOT, stated plainly
------------------------------
It is not a measured tyre cliff. Held-out **lap times do not support it**:
adding the term moves the lap-time calibration coefficient from 1.02 to 0.80
at ages 3-15, and refitting the slope around it drives the median slope
negative, which is impossible. That is what survivorship predicts - old-tyre
laps come only from tyres that behaved - but it is also what a wrong term
would produce, and lap times cannot tell the two apart. So the term absorbs
whatever makes long stints dearer in real races than in the linear
simulator: a cliff, tyre management, traffic behind fresher cars, or all of
them. Nothing downstream should call it degradation.

What it costs
-------------
After this calibration the sanity gate is **no longer fully independent** of
the strategy layer: the model has been fitted to a quantity that encodes how
good real strategists were. It keeps power over stop *timing* and compound
choice, which one global coefficient cannot tune, but it is not the
arm's-length check it was. Fit seasons and held-out seasons are therefore
reported separately.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("pitwall.models.degcurve")

# Cars whose extra stop was forced rather than chosen - a stint of five laps
# or fewer, or a stop inside the last five laps - are punctures, damage,
# penalties and late free stops. They are not strategy and would bias the
# moment towards "an extra stop is ruinous".
UNPLANNED_MIN_STINT = 5
UNPLANNED_LAST_STOP_FROM_END = 5

MIN_FINISHED_SHARE = 0.95
MAX_STOPS_CONSIDERED = 3


def _within(frame: pd.DataFrame, cols: list[str], key: str) -> pd.DataFrame:
    """Demean ``cols`` within ``key``: one-way fixed effects, no dummies."""
    dev = frame[cols].astype(float)
    return dev - dev.groupby(frame[key].values).transform("mean")


def _race_blocks(frame: pd.DataFrame, time_col: str) -> list[np.ndarray]:
    """One within-race demeaned (y, n_stops, grid) block per race.

    Demeaning inside the race is what makes this a fixed-effects estimate, and
    doing it once per race up front means the bootstrap can resample whole
    races by stacking blocks instead of rebuilding a frame each time.
    """
    f = frame.dropna(subset=[time_col, "n_stops", "grid", "race_id"])
    blocks = []
    for _, g in f.groupby("race_id", sort=False):
        if len(g) < 4 or g["n_stops"].nunique() < 2:
            continue
        m = g[[time_col, "n_stops", "grid"]].to_numpy(float)
        if not np.isfinite(m).all():
            continue
        m[:, 0] -= m[:, 0].min()  # gap to the winner of this race
        blocks.append(m - m.mean(axis=0))
    return blocks


def _beta_from_blocks(blocks: list[np.ndarray]) -> float:
    if len(blocks) < 3:
        return float("nan")
    m = np.vstack(blocks)
    if len(m) < 20:
        return float("nan")
    x, y = m[:, 1:], m[:, 0]
    try:
        return float(np.linalg.lstsq(x, y, rcond=None)[0][0])
    except np.linalg.LinAlgError:
        return float("nan")


def _beta_per_stop(frame: pd.DataFrame, time_col: str) -> float:
    """Seconds of finishing time per extra stop, with race fixed effects.

    Identified only by comparing cars against others in the same race, and
    controlling for where each car started.
    """
    return _beta_from_blocks(_race_blocks(frame, time_col))


def car_race_table(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """One row per classified car-race: real finishing time, stops, grid.

    Retirements are dropped, because a car that stopped on lap 30 did not run
    a strategy. So are the forced stops described at the top of this module.
    """
    lap_s = pd.to_numeric(laps["LapTime"], errors="coerce")
    d = laps.assign(lap_s=lap_s)
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


def observed_cost_of_extra_stop(
    laps: pd.DataFrame, stints: pd.DataFrame, n_boot: int = 400, seed: int = 0
) -> dict:
    """What one extra pit stop actually cost, in seconds of finishing time.

    The interval is a cluster bootstrap over races, because two cars in the
    same race are not independent observations of a strategy's value.
    """
    t = car_race_table(laps, stints)
    blocks = _race_blocks(t, "real_time_s")
    beta = _beta_from_blocks(blocks)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks)) if blocks else []
        b = _beta_from_blocks([blocks[i] for i in pick])
        if np.isfinite(b):
            boot.append(b)

    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return {
        "beta_s_per_stop": beta,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_car_races": int(sum(len(b) for b in blocks)),
        # Races where every car made the same number of stops carry no
        # information about what a stop costs, and are dropped rather than
        # contributing a column of zeros.
        "n_races": len(blocks),
    }


def simulated_cost_of_extra_stop(
    laps: pd.DataFrame,
    artefacts: dict,
    race_ids: list[str],
    curvature: float,
    n_sims: int = 200,
    seed: int = 7,
) -> dict:
    """The same quantity, computed from simulated finishing times.

    Every car keeps the strategy it actually ran; only the simulator's view of
    what those strategies cost changes with ``curvature``. The optimiser is
    not involved, so this does not calibrate the model against its own
    counterfactual.
    """
    from dataclasses import replace

    from pitwall import audit, pipeline
    from pitwall.sim import simulate

    rows = []
    cache: dict[str, object] = {}
    circuit_of = laps.groupby("race_id")["circuit"].first()

    for rid in race_ids:
        circuit = circuit_of.get(rid)
        if circuit is None:
            continue
        if circuit not in cache:
            cache[circuit] = pipeline.circuit_params(circuit, artefacts)
        params = cache[circuit]
        strategies = audit.reconstruct_strategies(laps, rid)
        if strategies.empty:
            continue
        race_laps = int(strategies["race_laps"].iloc[0])
        p = replace(
            params,
            race_laps=race_laps if race_laps >= 10 else params.race_laps,
            deg_quad_s_per_lap2=curvature,
        )
        field = audit.build_field(laps, rid, strategies)
        if len(field) < 6:
            continue
        res = simulate(p, field, n_sims=n_sims, seed=seed)
        st = strategies.reset_index(drop=True)
        for i, row in st.iterrows():
            if row["finished_share"] < MIN_FINISHED_SHARE:
                continue
            rows.append(
                {
                    "race_id": rid,
                    "car_id": row.get("car_id", f"{rid}_{row['driver']}"),
                    "sim_time_s": float(res.finish_times[:, i].mean()),
                    "n_stops": int(row["n_stops"]),
                    "grid": float(field[i].grid_position),
                }
            )

    t = pd.DataFrame(rows)
    if t.empty:
        return {"beta_s_per_stop": float("nan"), "n_car_races": 0}
    t = t[t["n_stops"].between(1, MAX_STOPS_CONSIDERED)]
    return {
        "beta_s_per_stop": _beta_per_stop(t, "sim_time_s"),
        "n_car_races": len(t),
        "n_races": int(t["race_id"].nunique()),
        "curvature": curvature,
    }


def solve_on_grid(target: float, curvatures: np.ndarray, moments: np.ndarray) -> float:
    """Curvature at which the simulated moment equals ``target``.

    ``np.interp`` needs its x-values ascending; the moment *falls* as
    curvature rises, so sort by the moment, not the curvature. Clipped to the
    grid rather than extrapolated.
    """
    order = np.argsort(moments)
    solved = float(np.interp(target, moments[order], curvatures[order]))
    return float(np.clip(solved, curvatures.min(), curvatures.max()))


def calibrate(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    artefacts: dict,
    fit_race_ids: list[str],
    target: float | None = None,
    grid: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004),
    n_sims: int = 200,
) -> dict:
    """Solve for the curvature that makes the model price a stop like reality.

    A coarse grid then linear interpolation: the moment is monotone in the
    curvature (more curvature makes long stints dearer, which makes an extra
    stop look better), and the moment itself carries a wide interval, so
    anything finer than this would be false precision.
    """
    if target is None:
        target = observed_cost_of_extra_stop(laps, stints)["beta_s_per_stop"]

    points = []
    for c in grid:
        got = simulated_cost_of_extra_stop(laps, artefacts, fit_race_ids, c, n_sims=n_sims)
        points.append((c, got["beta_s_per_stop"]))
        log.info(
            "curvature %.5f -> simulated cost per extra stop %+.2f s", c, got["beta_s_per_stop"]
        )

    pts = [(c, b) for c, b in points if np.isfinite(b)]
    if len(pts) < 2:
        return {"curvature": 0.0, "target": target, "points": points, "status": "no_solution"}

    cs = np.array([c for c, _ in pts])
    bs = np.array([b for _, b in pts])
    solved = solve_on_grid(target, cs, bs)

    return {
        "curvature": solved,
        "target": float(target),
        "points": points,
        "status": "ok",
        "at_grid_edge": bool(solved <= cs.min() + 1e-12 or solved >= cs.max() - 1e-12),
    }
