"""Strategy reconstruction, and the pre-registered sanity gate.

The reconstruction tests exist because getting this wrong is what failed the
sanity gate: inferring stops from FastF1's `Stint` counter invented pit stops
no team made, the simulator charged full pit loss for each, and the optimiser
"beat" real strategy by the cost of the phantom stops.
"""

from __future__ import annotations

import pandas as pd

from pitwall import audit


def _car_laps(in_laps, n_laps=50, ranks=None, driver="ABC"):
    """One car's race, with pit entries on the given laps."""
    ranks = ranks or {}
    rows = []
    rank = 0
    for lap in range(1, n_laps + 1):
        if lap in ranks:
            rank = ranks[lap]
        rows.append(
            {
                "race_id": "r1",
                "car_id": f"r1_{driver}",
                "Driver": driver,
                "Team": "T",
                "LapNumber": lap,
                "is_inlap": lap in in_laps,
                "compound_rank": rank,
                "TyreLife": 1,
                "Stint": 1,
            }
        )
    return pd.DataFrame(rows)


def test_a_one_stop_is_reconstructed_as_one_stop():
    laps = _car_laps(in_laps={20}, ranks={21: 2})
    s = audit.reconstruct_strategies(laps, "r1")
    assert len(s) == 1
    assert s["n_stops"].iloc[0] == 1
    assert s["strategy"].iloc[0].stops == ((20, 2),)


def test_consecutive_in_laps_are_one_stop_not_two():
    """The artefact that broke the audit: two in-laps a lap apart."""
    laps = _car_laps(in_laps={35, 36}, ranks={36: 2})
    s = audit.reconstruct_strategies(laps, "r1")
    assert s["n_stops"].iloc[0] == 1


def test_three_consecutive_in_laps_collapse_to_one():
    laps = _car_laps(in_laps={35, 36, 37}, ranks={36: 2})
    s = audit.reconstruct_strategies(laps, "r1")
    assert s["n_stops"].iloc[0] == 1


def test_a_stop_on_the_final_lap_is_ignored():
    """It changes nothing about the race and would cost a phantom pit loss."""
    laps = _car_laps(in_laps={50}, n_laps=50)
    s = audit.reconstruct_strategies(laps, "r1")
    assert s["n_stops"].iloc[0] == 0


def test_genuine_two_stop_survives():
    laps = _car_laps(in_laps={15, 35}, ranks={16: 1, 36: 2})
    s = audit.reconstruct_strategies(laps, "r1")
    assert s["strategy"].iloc[0].stops == ((15, 1), (35, 2))


def test_implausible_stop_count_is_dropped_not_audited():
    laps = _car_laps(in_laps={5, 12, 19, 26, 33, 40})
    s = audit.reconstruct_strategies(laps, "r1")
    assert s.empty, "a six-stop reconstruction should be dropped, not modelled"


def test_stint_counter_alone_does_not_create_stops():
    """FastF1's Stint column increments for reasons other than a pit stop."""
    laps = _car_laps(in_laps=set())
    laps.loc[laps["LapNumber"] > 25, "Stint"] = 2  # counter moves, no in-lap
    s = audit.reconstruct_strategies(laps, "r1")
    assert s["n_stops"].iloc[0] == 0


# --------------------------------------------------------------------------
# The gate itself
# --------------------------------------------------------------------------


def _audit_frame(gains):
    return pd.DataFrame(
        {
            "gain_s": gains,
            "race_id": ["r1"] * len(gains),
            "team": ["T"] * len(gains),
            "driver": ["D"] * len(gains),
        }
    )


def test_gate_passes_when_the_optimiser_barely_beats_reality():
    g = audit.sanity_gate(_audit_frame([0.0, 0.1, 0.3, 0.0, 0.2]))
    assert g["passed"], g["failures"]


def test_gate_fails_on_an_implausible_mean_gain():
    g = audit.sanity_gate(_audit_frame([20.0] * 10))
    assert not g["passed"]
    assert any("mean gain" in f for f in g["failures"])


def test_gate_fails_when_the_optimiser_beats_almost_everyone():
    g = audit.sanity_gate(_audit_frame([0.5] * 100))
    assert not g["passed"]
    assert any("car-races" in f for f in g["failures"])


def test_gate_fails_on_a_single_absurd_gain():
    g = audit.sanity_gate(_audit_frame([0.0] * 99 + [90.0]))
    assert not g["passed"]
    assert any("largest single gain" in f for f in g["failures"])


def test_gate_thresholds_are_the_pre_registered_ones():
    g = audit.sanity_gate(_audit_frame([0.0, 0.1]))
    assert g["thresholds"] == {
        "max_mean_gain_s": 2.0,
        "max_share_improved": 0.70,
        "max_single_gain_s": 30.0,
    }


def test_start_compound_is_held_fixed_by_default():
    """A counterfactual that changes the start tyre answers the wrong question.

    From 2018-2021 the top ten had to start on their Q2 tyre. An optimiser
    free to ignore that beats the team by breaking a rule the team obeyed.
    """
    cfg = audit.AuditConfig()
    assert cfg.fix_start_compound is True
