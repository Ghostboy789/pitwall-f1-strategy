"""Validation V6: the strategy-level moments must recover what is put in."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.models import strategy_validation as sv


def _field(stop_cost: float, balance_cost: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for race in range(30):
        for car in range(16):
            n_stops = 1 + car % 2
            share = 0.5 + 0.3 * rng.random()
            grid = car + 1
            t = 5000 + 1.5 * grid + stop_cost * n_stops + balance_cost * share
            rows.append(
                {
                    "race_id": f"r{race}",
                    "n_stops": n_stops,
                    "longest_stint_share": share,
                    "grid": float(grid),
                    "time": t + rng.normal(0, 2) + race * 7.0,
                }
            )
    return pd.DataFrame(rows)


def test_moments_recover_known_costs():
    m = sv.moments(_field(stop_cost=20.0, balance_cost=100.0), "time", n_boot=200)
    assert abs(m["extra_stop_s"]["estimate"] - 20.0) < 1.5
    assert abs(m["stint_balance_s"]["estimate"] - 100.0) < 10.0
    assert m["extra_stop_s"]["ci_lo"] < 20.0 < m["extra_stop_s"]["ci_hi"]


def test_moments_see_nothing_when_strategy_is_free():
    m = sv.moments(_field(stop_cost=0.0, balance_cost=0.0), "time", n_boot=200)
    for key in ("extra_stop_s", "stint_balance_s"):
        assert m[key]["ci_lo"] < 0.0 < m[key]["ci_hi"]


def test_car_table_drops_forced_stops():
    """A car that pitted on lap 2 is not running a strategy."""
    laps, stints = [], []
    for race in ("r1", "r2", "r3"):
        for i in range(8):
            for lap in range(1, 51):
                laps.append(
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
            spans = [(1, 2), (3, 26), (27, 50)] if i == 0 else [(1, 25), (26, 50)]
            for k, (a, b) in enumerate(spans):
                stints.append(
                    {
                        "race_id": race,
                        "car_id": f"{race}_{i}",
                        "stint_length": float(b - a + 1),
                        "end_lap": float(b),
                        "ended_in_pit": k < len(spans) - 1,
                    }
                )
    t = sv.car_race_table(pd.DataFrame(laps), pd.DataFrame(stints))
    assert len(t) == 21
    assert not t["car_id"].str.endswith("_0").any()
