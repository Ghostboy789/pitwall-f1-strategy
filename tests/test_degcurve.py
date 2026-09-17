"""The curvature term, and the moment it is pinned on.

The point of the curvature is that a linear degradation model under-prices
long stints, so the optimiser prefers one fewer stop than teams made. These
check that adding curvature does what it is supposed to do, and that zero
curvature reproduces the old linear behaviour exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.models import degcurve
from pitwall.optimize import deterministic_cost
from pitwall.sim import Car, CircuitParams, Strategy, simulate


def _params(**kw) -> CircuitParams:
    base = dict(
        circuit="test",
        race_laps=60,
        base_lap_s=90.0,
        pit_loss_s=22.0,
        fuel_s_per_lap=-0.06,
        caution_hazard_per_lap=0.0,
        traffic_s=1.0,
        deg_by_rank={0: 0.04, 1: 0.03, 2: 0.02},
        pass_p_base=0.08,
        lap_time_sd_s=0.0,
    )
    base.update(kw)
    return CircuitParams(**base)


def test_zero_curvature_is_the_old_linear_model():
    """The default must not move any existing number."""
    a, _ = deterministic_cost((30,), (0, 1), 60, {0: 0.04, 1: 0.03}, pit_loss_s=22.0)
    b, _ = deterministic_cost(
        (30,), (0, 1), 60, {0: 0.04, 1: 0.03}, pit_loss_s=22.0, deg_quad_s_per_lap2=0.0
    )
    assert a == b


def test_curvature_matches_the_hand_calculation():
    """Sum of c*a^2 over a=1..n is c*n(n+1)(2n+1)/6, twice for two stints."""
    c = 0.002
    cost, lengths = deterministic_cost(
        (30,), (0, 0), 60, {0: 0.0}, pit_loss_s=0.0, deg_quad_s_per_lap2=c
    )
    assert lengths == (30, 30)
    expected = 2 * c * 30 * 31 * 61 / 6.0
    assert cost == np.float64(expected) or abs(cost - expected) < 1e-9


def test_curvature_makes_the_extra_stop_worth_taking():
    """The whole reason the term exists.

    With a straight line, one long stint beats two short ones here. Curvature
    prices the back end of a long stint properly and flips the answer.
    """
    deg = {0: 0.04, 1: 0.04}
    one_stop, _ = deterministic_cost((30,), (0, 1), 60, deg, pit_loss_s=22.0)
    two_stop, _ = deterministic_cost((20, 40), (0, 1, 0), 60, deg, pit_loss_s=22.0)
    assert one_stop < two_stop, "linear model should prefer the one-stop here"

    one_c, _ = deterministic_cost(
        (30,), (0, 1), 60, deg, pit_loss_s=22.0, deg_quad_s_per_lap2=0.003
    )
    two_c, _ = deterministic_cost(
        (20, 40), (0, 1, 0), 60, deg, pit_loss_s=22.0, deg_quad_s_per_lap2=0.003
    )
    assert two_c < one_c, "curvature should make the extra stop worth taking"


def test_simulator_honours_the_curvature():
    """A curved tyre must be slower over a long stint, and by the right order."""
    car = [
        Car(
            driver="A",
            pace_offset_s=0.0,
            grid_position=1,
            strategy=Strategy(stops=((30, 1),), start_rank=0),
            team="T",
        )
    ]
    flat = simulate(_params(), car, n_sims=40, seed=3).finish_times.mean()
    curved = simulate(
        _params(deg_quad_s_per_lap2=0.002), car, n_sims=40, seed=3
    ).finish_times.mean()
    assert curved > flat
    # two stints of 30: 2 * 0.002 * 30*31*61/6 = 37.8s
    assert 30 < curved - flat < 45


def test_solver_interpolates_a_falling_moment():
    """The real grid from the first calibration run, which the original
    solver got wrong by returning the grid edge."""
    cs = np.array([0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004])
    bs = np.array([10.28, 6.01, 2.39, -1.00, -4.57, -12.37, -20.61])
    got = degcurve.solve_on_grid(1.08, cs, bs)
    assert 0.001 < got < 0.0015
    assert abs(got - 0.0011932) < 1e-5


def test_observed_moment_is_computable_and_drops_forced_stops():
    """A car that pitted on lap 2 is not running a strategy."""
    rows = []
    for race in ("r1", "r2", "r3", "r4"):
        for i in range(10):
            rows.append(
                {
                    "race_id": race,
                    "circuit": "c",
                    "car_id": f"{race}_{i}",
                    "LapNumber": 1,
                    "LapTime": 90.0,
                    "Position": float(i + 1),
                    "race_laps": 50.0,
                }
            )
            for lap in range(2, 51):
                rows.append(
                    {
                        "race_id": race,
                        "circuit": "c",
                        "car_id": f"{race}_{i}",
                        "LapNumber": lap,
                        "LapTime": 90.0 + i * 0.1,
                        "Position": float(i + 1),
                        "race_laps": 50.0,
                    }
                )
    laps = pd.DataFrame(rows)

    st = []
    for race in ("r1", "r2", "r3", "r4"):
        for i in range(10):
            forced = i == 0  # one car per race pits on lap 2
            if forced:
                spans = [(1, 2), (3, 26), (27, 50)]
            else:
                spans = [(1, 25), (26, 50)] if i % 2 else [(1, 17), (18, 34), (35, 50)]
            for k, (a, b) in enumerate(spans):
                st.append(
                    {
                        "race_id": race,
                        "car_id": f"{race}_{i}",
                        "stint_length": float(b - a + 1),
                        "start_lap": float(a),
                        "end_lap": float(b),
                        "ended_in_pit": k < len(spans) - 1,
                        "race_laps": 50.0,
                    }
                )
    stints = pd.DataFrame(st)

    t = degcurve.car_race_table(laps, stints)
    assert not t.empty
    assert not any(t["car_id"].str.endswith("_0")), "forced-stop cars must be dropped"

    out = degcurve.observed_cost_of_extra_stop(laps, stints, n_boot=20)
    assert np.isfinite(out["beta_s_per_stop"])
    assert out["n_races"] == 4
