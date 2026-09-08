"""The headline finding: what track position is worth, in seconds, per circuit.

The question
------------
The same tyre advantage buys a completely different outcome at Monza than at
Monaco. An undercut worth three seconds and a place is worth three seconds at
Monza, because you keep the place. At Monaco it can be worth nothing, because
the car you jumped is still ahead on the road and you are not getting past.

Everything else in this project is machinery in service of putting a number on
that difference instead of asserting it.

The definition
--------------
For a car with a pace advantage of ``delta`` seconds per lap over the car
directly ahead:

    per-lap probability of completing the pass   p       (from the model)
    expected laps spent stuck                    min(1/p, laps_remaining)
    time lost while stuck                        delta per lap

    value of track position = delta * min(1/p, laps_remaining)

That is the expected race-time cost of being behind rather than ahead. At a
circuit where p is near zero the faster car is stuck for the rest of the race
and the cost is the whole remaining deficit; where p is high it costs a lap or
two of being held up.

The geometric expectation ``1/p`` assumes each lap is an independent attempt.
That is an approximation - a car that has failed to pass for ten laps is
probably in a worse position than the average attacker, not the same one - and
it makes this estimate, if anything, optimistic about escaping. Stated here
rather than buried.

Uncertainty
-----------
Reported as a cluster bootstrap over **races**, not over rows. Opportunities
within one race share weather, track state and the same cars, so resampling
rows would treat 300 correlated observations as 300 independent ones and
produce intervals far too narrow. The number of races behind each circuit's
estimate is carried alongside it everywhere it appears.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pitwall import config
from pitwall.models import overtaking as ot

log = logging.getLogger("pitwall.trackposition")

# The standard scenario every circuit is evaluated at, fixed before any
# results were computed. A meaningful but not overwhelming pace advantage, a
# car actively attacking within DRS range, at mid-race distance.
SCENARIO = {
    "pace_delta_s": -0.5,  # follower is 0.5 s/lap faster (negative = quicker)
    "gap_s": 1.0,  # within DRS range, actively attacking
    "drs_available": 1,
    "tyre_age_delta": -10.0,  # fresher tyre, the usual reason for the advantage
    "compound_rank_delta": -1.0,
    "lap_progress": 0.5,
    "position": 8,  # midfield, where most racing happens
}

DEFAULT_LAPS_REMAINING = 25
MIN_RACES_FOR_ESTIMATE = 2
N_BOOTSTRAP = 120


def scenario_frame(
    circuits: list[str], laps_remaining: int = DEFAULT_LAPS_REMAINING, **overrides
) -> pd.DataFrame:
    """One row per circuit, all at the identical standard scenario."""
    from pitwall.circuits import CIRCUIT_REF

    s = {**SCENARIO, **overrides}
    rows = []
    for c in circuits:
        ref = CIRCUIT_REF.get(c)
        rows.append(
            {
                "circuit": c,
                **s,
                "drs_zones": ref.drs_zones if ref else np.nan,
                "street": int(ref.street) if ref else np.nan,
                "laps_remaining": laps_remaining,
            }
        )
    return pd.DataFrame(rows)


def value_from_p(p: np.ndarray, delta_s: float, laps_remaining: int) -> np.ndarray:
    """Seconds of race time lost, given a per-lap pass probability."""
    p = np.clip(p, 1e-6, 1.0)
    laps_stuck = np.minimum(1.0 / p, laps_remaining)
    return abs(delta_s) * laps_stuck


def _fit_predict(train: pd.DataFrame, scen: pd.DataFrame, seed: int):
    """Fit the hierarchical pass model on `train` and score the scenario.

    The hierarchical form is essential here, not a refinement. Without a
    per-circuit term the model reaches a circuit only through its DRS-zone
    count and whether it is a street track, so every circuit sharing that pair
    receives an identical probability - and a per-circuit headline number
    computed from it would be an artefact of that grouping rather than a
    finding.
    """
    fitted = ot.fit_hierarchical(train, seed=seed)
    return ot.predict_hierarchical(fitted, scen)


def estimate(
    opps: pd.DataFrame,
    laps_remaining: int = DEFAULT_LAPS_REMAINING,
    n_boot: int = N_BOOTSTRAP,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Value of track position per circuit, with a cluster-bootstrap interval."""
    d = opps.dropna(subset=["pace_delta_s"]).copy()

    counts = d.groupby("circuit")["race_id"].nunique()
    circuits = sorted(counts[counts >= MIN_RACES_FOR_ESTIMATE].index)
    dropped = sorted(set(counts.index) - set(circuits))
    if dropped:
        log.info("circuits excluded for having < %d races: %s", MIN_RACES_FOR_ESTIMATE, dropped)

    scen = scenario_frame(circuits, laps_remaining)
    point_p = _fit_predict(d, scen, seed)

    # Cluster bootstrap over races, STRATIFIED BY CIRCUIT.
    #
    # An unstratified resample draws races from the whole calendar, so a
    # circuit with three races frequently receives none at all - its effect
    # then reverts to the global mean and its value lands mid-table. That
    # produced intervals spanning the entire range for every circuit and made
    # nothing distinguishable from anything.
    #
    # It also answers the wrong question. The uncertainty worth reporting is
    # "how well do three races at Monaco pin down Monaco?", not "what if the
    # calendar had visited different circuits?". Resampling races *within*
    # each circuit, preserving that circuit's race count, answers the first.
    rng = np.random.default_rng(seed)
    races_by_circuit = {
        c: d.loc[d["circuit"] == c, "race_id"].unique() for c in d["circuit"].unique()
    }
    by_race = {r: g for r, g in d.groupby("race_id")}
    boot = np.full((n_boot, len(circuits)), np.nan)
    for b in range(n_boot):
        pick = np.concatenate(
            [rng.choice(rs, size=len(rs), replace=True) for rs in races_by_circuit.values()]
        )
        sample = pd.concat([by_race[r] for r in pick], ignore_index=True)
        if sample["passed"].nunique() < 2:
            continue
        try:
            boot[b] = _fit_predict(sample, scen, seed + b)
        except Exception:
            continue
        if (b + 1) % 20 == 0:
            log.info("bootstrap %d/%d", b + 1, n_boot)

    delta = SCENARIO["pace_delta_s"]
    out = pd.DataFrame(
        {
            "circuit": circuits,
            "n_races": [int(counts[c]) for c in circuits],
            "n_opportunities": [int((d["circuit"] == c).sum()) for c in circuits],
            "observed_pass_rate": [
                float(d.loc[d["circuit"] == c, "passed"].mean()) for c in circuits
            ],
            "p_pass_per_lap": point_p,
            "value_s": value_from_p(point_p, delta, laps_remaining),
            # Uncapped, so the ceiling is visible rather than silent. Where
            # this exceeds laps_remaining the car is stuck for the rest of the
            # race and `value_s` is pinned at the cap -- which is itself the
            # finding, not a limitation of the arithmetic.
            "expected_laps_stuck": 1.0 / np.clip(point_p, 1e-6, 1.0),
            "at_ceiling": (1.0 / np.clip(point_p, 1e-6, 1.0)) >= laps_remaining,
        }
    )

    vals = value_from_p(boot, delta, laps_remaining)
    out["p_lo"] = np.nanpercentile(boot, 2.5, axis=0)
    out["p_hi"] = np.nanpercentile(boot, 97.5, axis=0)
    # A lower pass probability means a HIGHER cost, so the interval flips.
    out["value_lo"] = np.nanpercentile(vals, 2.5, axis=0)
    out["value_hi"] = np.nanpercentile(vals, 97.5, axis=0)
    out["n_boot_ok"] = int(np.isfinite(boot).all(axis=1).sum())
    out["laps_remaining"] = laps_remaining
    out["pace_delta_s"] = delta

    return out.sort_values("value_s", ascending=False).reset_index(drop=True)


def separation_test(est: pd.DataFrame, a: str, b: str) -> dict:
    """Are two circuits' track-position values actually distinguishable?

    The validation plan commits to a falsification condition: if the
    per-circuit values cannot be told apart, the project's central claim is
    refuted. This is the test, applied to a named pair.
    """
    ra = est[est["circuit"] == a]
    rb = est[est["circuit"] == b]
    if ra.empty or rb.empty:
        return {"status": "circuit_missing", "a": a, "b": b}
    ra, rb = ra.iloc[0], rb.iloc[0]
    overlap = not (ra["value_lo"] > rb["value_hi"] or rb["value_lo"] > ra["value_hi"])
    return {
        "a": a,
        "b": b,
        "value_a": float(ra["value_s"]),
        "ci_a": (float(ra["value_lo"]), float(ra["value_hi"])),
        "value_b": float(rb["value_s"]),
        "ci_b": (float(rb["value_lo"]), float(rb["value_hi"])),
        "ratio": float(ra["value_s"] / rb["value_s"]) if rb["value_s"] else np.nan,
        "intervals_overlap": bool(overlap),
        "distinguishable": bool(not overlap),
    }
