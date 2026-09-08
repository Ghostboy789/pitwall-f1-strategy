"""The simulator's behaviour must follow from its mechanics, not from tuning.

The tests that matter here are the ones about *blocking*. If a faster car can
always get past, track position is worth nothing and the whole project has no
subject. These pin that the clamp works and that it responds to the circuit.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitwall.sim import Car, CircuitParams, Strategy, compare_strategies, simulate


def _params(pass_p_base: float, **kw) -> CircuitParams:
    base = dict(
        circuit="test",
        race_laps=30,
        base_lap_s=90.0,
        pit_loss_s=22.0,
        fuel_s_per_lap=-0.06,
        caution_hazard_per_lap=0.0,
        traffic_s=1.0,
        deg_by_rank={0: 0.08, 1: 0.05, 2: 0.03},
        pass_p_base=pass_p_base,
        lap_time_sd_s=0.0,
    )
    base.update(kw)
    return CircuitParams(**base)


def _field(n=6, pace_spread=0.3):
    return [
        Car(
            driver=f"D{i}",
            pace_offset_s=i * pace_spread,
            grid_position=i + 1,
            strategy=Strategy(stops=((15, 2),), start_rank=0),
        )
        for i in range(n)
    ]


def test_faster_car_wins_when_passing_is_easy():
    """With passing near-certain, pace order should decide the race."""
    res = simulate(_params(0.9), _field(), n_sims=200, seed=1)
    mean_pos = res.finish_positions.mean(axis=0)
    assert mean_pos[0] < mean_pos[-1], "the quickest car should finish ahead"


def test_blocking_preserves_grid_order_when_passing_is_impossible():
    """The core mechanic. If nobody can pass, a faster car stays stuck."""
    cars = [
        Car(
            driver="slow_ahead",
            pace_offset_s=1.0,
            grid_position=1,
            strategy=Strategy(stops=(), start_rank=2),
        ),
        Car(
            driver="fast_behind",
            pace_offset_s=0.0,
            grid_position=2,
            strategy=Strategy(stops=(), start_rank=2),
        ),
    ]
    res = simulate(_params(1e-6), cars, n_sims=200, seed=2)
    stuck_rate = (res.finish_positions[:, 1] == 2).mean()
    assert stuck_rate > 0.95, f"faster car escaped {1 - stuck_rate:.0%} of the time"


def test_same_car_escapes_when_passing_is_easy():
    """Identical setup, easy circuit: now the faster car gets through."""
    cars = [
        Car(
            driver="slow_ahead",
            pace_offset_s=1.0,
            grid_position=1,
            strategy=Strategy(stops=(), start_rank=2),
        ),
        Car(
            driver="fast_behind",
            pace_offset_s=0.0,
            grid_position=2,
            strategy=Strategy(stops=(), start_rank=2),
        ),
    ]
    res = simulate(_params(0.8), cars, n_sims=200, seed=3)
    assert (res.finish_positions[:, 1] == 1).mean() > 0.9


def test_track_position_value_is_circuit_dependent():
    """The project's central claim, in miniature.

    The same pace advantage must be worth more where passing is hard. If this
    fails, the simulator cannot express the finding it exists to produce.
    """
    cars = [
        Car(
            driver="ahead",
            pace_offset_s=0.6,
            grid_position=1,
            strategy=Strategy(stops=(), start_rank=2),
        ),
        Car(
            driver="behind",
            pace_offset_s=0.0,
            grid_position=2,
            strategy=Strategy(stops=(), start_rank=2),
        ),
    ]
    hard = simulate(_params(0.01), cars, n_sims=300, seed=4)
    easy = simulate(_params(0.5), cars, n_sims=300, seed=4)
    loss_hard = hard.gaps_to_winner[:, 1].mean()
    loss_easy = easy.gaps_to_winner[:, 1].mean()
    assert loss_hard > loss_easy, "being stuck must cost more where passing is hard"


def test_safety_car_makes_a_stop_cheaper():
    """A stop under caution must cost less, or the biggest strategic lever is missing."""
    cars = _field(n=4)
    no_sc = simulate(_params(0.3, caution_hazard_per_lap=0.0), cars, n_sims=300, seed=5)
    with_sc = simulate(_params(0.3, caution_hazard_per_lap=0.08), cars, n_sims=300, seed=5)
    assert with_sc.n_cautions.mean() > 0
    assert no_sc.n_cautions.sum() == 0


def test_output_is_a_distribution_not_a_point():
    res = simulate(_params(0.2, lap_time_sd_s=0.3), _field(), n_sims=150, seed=6)
    assert res.finish_positions.shape == (150, 6)
    assert res.finish_positions.std(axis=0).sum() > 0, "no variation across simulations"


def test_positions_are_a_valid_permutation_every_sim():
    res = simulate(_params(0.3, lap_time_sd_s=0.3), _field(n=8), n_sims=100, seed=7)
    expected = set(range(1, 9))
    for row in res.finish_positions:
        assert set(row.tolist()) == expected


def test_seed_is_reproducible():
    a = simulate(_params(0.3, lap_time_sd_s=0.3), _field(), n_sims=100, seed=42)
    b = simulate(_params(0.3, lap_time_sd_s=0.3), _field(), n_sims=100, seed=42)
    assert np.array_equal(a.finish_positions, b.finish_positions)


def test_compare_strategies_holds_the_rest_of_the_field_fixed():
    cars = _field(n=5)
    out = compare_strategies(
        _params(0.3),
        cars,
        focus_index=2,
        alternatives=[Strategy(stops=((12, 2),)), Strategy(stops=((20, 2),))],
        n_sims=120,
    )
    assert len(out) == 2
    assert {"mean_position", "gap_lo", "gap_hi"} <= set(out.columns)


def test_strategy_compound_lookup():
    s = Strategy(stops=((10, 1), (25, 2)), start_rank=0)
    assert s.compound_at(1) == 0
    assert s.compound_at(10) == 0
    assert s.compound_at(11) == 1
    assert s.compound_at(26) == 2


@pytest.mark.parametrize("p", [0.001, 0.05, 0.5])
def test_pass_probability_increases_with_pace_advantage(p):
    params = _params(p)
    probs = params.pass_probability(np.array([0.0, 0.5, 2.0]))
    assert probs[0] <= probs[1] <= probs[2]
    assert np.all((probs >= 0) & (probs <= 1))
