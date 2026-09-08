"""Trap T2/T5: the degradation estimator must recover a slope it was given.

These are the tests that would have caught the bug this module actually had.
Centring within a stint made tyre age and fuel burn the identical vector, so
the estimator returned (degradation - fuel) and every real-world cell came out
negative. ``test_recovers_known_slope_with_fuel_confound`` fails under that
specification and passes under the current one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.models.degradation import _within_car_centred, _wls_multi


def _synthetic_race(deg: float, fuel: float, n_cars: int = 12, seed: int = 0) -> pd.DataFrame:
    """A race with a known degradation slope and a known fuel effect.

    Each car runs two stints of differing length, which is what makes tyre age
    and laps completed separable at all.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for car in range(n_cars):
        pit = 15 + car % 9  # stops spread across laps 15-23
        base = 90.0 + car * 0.15  # cars differ in pace
        for lap in range(1, 51):
            age = lap if lap <= pit else lap - pit
            t = base + deg * age + fuel * (lap - 1) + rng.normal(0, 0.05)
            rows.append(
                {
                    "car_id": f"c{car}",
                    "stint_id": f"c{car}_s{1 if lap <= pit else 2}",
                    "LapNumber": lap,
                    "lap_time_s": t,
                    "TyreLife": age,
                    "tyre_age": age,
                    "fuel_laps_burned": lap - 1,
                    "circuit": "test",
                    "compound_rank_label": "MIDDLE",
                }
            )
    return pd.DataFrame(rows)


def test_recovers_known_slope_with_fuel_confound():
    """The estimator must return the degradation it was given, not deg - fuel."""
    deg, fuel = 0.08, -0.06
    d = _within_car_centred(_synthetic_race(deg, fuel))
    x = np.column_stack([d["tyre_age_dev"], d["fuel_dev"]])
    beta, se, n, vif = _wls_multi(x, d["lap_time_dev"].to_numpy(float), np.ones(len(d)))
    assert abs(beta[0] - deg) < 0.01, f"degradation {beta[0]:.4f} != {deg}"
    assert abs(beta[1] - fuel) < 0.01, f"fuel {beta[1]:.4f} != {fuel}"
    assert vif < 5, f"tyre age and fuel are not separated (VIF {vif:.1f})"


def test_ignoring_fuel_biases_the_slope_downward():
    """Documents the bug: omit the fuel control and the slope is deg + fuel."""
    deg, fuel = 0.08, -0.06
    d = _within_car_centred(_synthetic_race(deg, fuel))
    x = d[["tyre_age_dev"]].to_numpy(float)
    beta, _, _, _ = _wls_multi(x, d["lap_time_dev"].to_numpy(float), np.ones(len(d)))
    assert beta[0] < deg, "omitting fuel must bias the slope downward"


def test_positive_degradation_stays_positive():
    """A real tyre never gets faster with age; a correct estimator says so."""
    d = _within_car_centred(_synthetic_race(0.10, -0.06))
    x = np.column_stack([d["tyre_age_dev"], d["fuel_dev"]])
    beta, _, _, _ = _wls_multi(x, d["lap_time_dev"].to_numpy(float), np.ones(len(d)))
    assert beta[0] > 0


def test_weights_are_respected():
    """Weighted and unweighted fits differ when the weights are not uniform."""
    d = _within_car_centred(_synthetic_race(0.08, -0.06))
    x = np.column_stack([d["tyre_age_dev"], d["fuel_dev"]])
    y = d["lap_time_dev"].to_numpy(float)
    w = np.where(d["tyre_age_dev"] > 0, 3.0, 1.0)
    b_flat, _, _, _ = _wls_multi(x, y, np.ones(len(d)))
    b_wt, _, _, _ = _wls_multi(x, y, w)
    assert not np.allclose(b_flat, b_wt)


def test_too_few_rows_returns_nan_not_a_wrong_answer():
    x = np.ones((5, 2))
    beta, se, n, _ = _wls_multi(x, np.ones(5), np.ones(5))
    assert np.isnan(beta).all()
    assert n == 5
