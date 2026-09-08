"""The headline metric's arithmetic, and the pass detector's core rule.

Two things are pinned here. First, that the value of track position responds
to the pass probability in the direction and shape the finding claims. Second,
that a position swap bought in the pit lane is never counted as an overtake --
without that exclusion Monaco would look as passable as Monza and the entire
project would say nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall import trackposition as tp
from pitwall.models import overtaking as ot

# --------------------------------------------------------------------------
# The headline metric
# --------------------------------------------------------------------------


def test_value_falls_as_passing_gets_easier():
    p = np.array([0.005, 0.05, 0.25, 0.9])
    v = tp.value_from_p(p, delta_s=-0.5, laps_remaining=25)
    assert np.all(np.diff(v) < 0), "an easier circuit must cost less"


def test_value_is_capped_by_laps_remaining():
    """A car cannot lose more than the rest of the race."""
    v = tp.value_from_p(np.array([1e-9]), delta_s=-0.5, laps_remaining=25)
    assert v[0] == 25 * 0.5


def test_value_scales_with_pace_advantage():
    a = tp.value_from_p(np.array([0.05]), delta_s=-0.5, laps_remaining=40)
    b = tp.value_from_p(np.array([0.05]), delta_s=-1.0, laps_remaining=40)
    assert np.isclose(b[0], 2 * a[0])


def test_value_matches_the_closed_form_by_hand():
    # p = 0.05 -> 20 laps stuck, under a 40-lap ceiling, at 0.5 s/lap.
    v = tp.value_from_p(np.array([0.05]), delta_s=-0.5, laps_remaining=40)
    assert np.isclose(v[0], 10.0)


def test_scenario_is_identical_for_every_circuit():
    """The comparison is only meaningful if nothing but the circuit changes."""
    f = tp.scenario_frame(["monaco", "monza"], laps_remaining=25)
    varying = [c for c in tp.SCENARIO if f[c].nunique() > 1]
    assert not varying, f"scenario differs across circuits in {varying}"


def test_separation_test_reports_overlap_honestly():
    est = pd.DataFrame(
        {
            "circuit": ["a", "b"],
            "value_s": [10.0, 2.0],
            "value_lo": [8.0, 1.0],
            "value_hi": [12.0, 3.0],
        }
    )
    r = tp.separation_test(est, "a", "b")
    assert r["distinguishable"] is True
    assert np.isclose(r["ratio"], 5.0)

    est.loc[1, "value_hi"] = 9.0  # now the intervals overlap
    r2 = tp.separation_test(est, "a", "b")
    assert r2["distinguishable"] is False


# --------------------------------------------------------------------------
# The pass detector
# --------------------------------------------------------------------------


def _two_car_race(pit_on_lap: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two cars; the follower moves ahead between lap 1 and lap 2."""
    rows = []
    for lap in (1, 2):
        for pos, drv in ((1, "A"), (2, "B")) if lap == 1 else ((1, "B"), (2, "A")):
            rows.append(
                {
                    "race_id": "r1",
                    "circuit": "monza",
                    "era": "2022-2025_ground_effect",
                    "year": 2023,
                    "Driver": drv,
                    "Team": "T",
                    "LapNumber": lap,
                    "Position": pos,
                    "Time": 90.0 * lap + pos * 0.5,
                    "TyreLife": 5,
                    "Compound": "MEDIUM",
                    "compound_rank": 1,
                    "is_inlap": bool(pit_on_lap == lap and drv == "B"),
                    "is_outlap": bool(
                        pit_on_lap is not None and lap == pit_on_lap + 1 and drv == "B"
                    ),
                    "is_green": True,
                    "race_laps": 50,
                    "gap_ahead_s": 0.5,
                }
            )
    laps = pd.DataFrame(rows)
    pace = laps.assign(lap_time_s=90.0, fuel_laps_burned=laps["LapNumber"] - 1)
    pace["car_id"] = pace["race_id"] + "_" + pace["Driver"]
    return laps, pace


def test_a_racing_pass_is_detected():
    laps, pace = _two_car_race()
    opps = ot.build_opportunities(laps, pace)
    assert len(opps) == 1
    assert opps["passed"].iloc[0] == 1


def test_a_pit_stop_swap_is_not_an_overtake():
    """The single most important exclusion in the detector."""
    laps, pace = _two_car_race(pit_on_lap=1)
    opps = ot.build_opportunities(laps, pace)
    assert len(opps) == 0, "a position swap bought in the pit lane was counted as a pass"


def test_cars_beyond_striking_distance_are_not_opportunities():
    laps, pace = _two_car_race()
    laps.loc[laps["Position"] == 2, "Time"] += 10.0  # 10s back, not attacking
    opps = ot.build_opportunities(laps, pace)
    assert len(opps) == 0


def test_detector_sanity_orders_circuits_by_passes_per_race():
    opps = pd.DataFrame(
        {
            "circuit": ["monaco"] * 4 + ["monza"] * 4,
            "race_id": ["a", "a", "b", "b"] * 2,
            "passed": [0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    s = ot.detector_sanity(opps).set_index("circuit")
    assert s.loc["monaco", "mean_passes_per_race"] < s.loc["monza", "mean_passes_per_race"]


def test_reliability_is_perfect_for_a_perfect_model():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.9, 4000)
    y = (rng.random(4000) < p).astype(int)
    rel = ot.reliability(y, p, n_bins=10)
    assert rel.attrs["ece"] < 0.03, f"ECE {rel.attrs['ece']:.4f} too high for a calibrated model"
    assert (rel["n"] > 0).all()
