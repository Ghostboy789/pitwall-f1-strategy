"""Data-quality gates (V5) and the pre-registered exclusion rules (E1-E5).

These are the checks that stop a pipeline bug from becoming a published
number, so they are tested against deliberately broken frames rather than only
against good ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall import quality


def _laps(n_cars=3, n_laps=20, circuit="monza"):
    rows = []
    for c in range(n_cars):
        for lap in range(1, n_laps + 1):
            stint = 1 if lap <= 10 else 2
            rows.append(
                {
                    "car_id": f"r1_D{c}",
                    "race_id": "r1",
                    "stint_id": f"r1_D{c}_s{stint}",
                    "Driver": f"D{c}",
                    "LapNumber": lap,
                    "Stint": stint,
                    "TyreLife": lap if stint == 1 else lap - 10,
                    "LapTime": 90.0,
                    "lap_time_s": 90.0,
                    "TrackStatus": "1",
                    "is_green": True,
                    "is_inlap": lap == 10,
                    "is_outlap": lap == 11,
                    "is_wet_tyre": False,
                    "FreshTyre": True,
                    "circuit": circuit,
                    "race_laps": n_laps,
                    "year": 2023,
                    "round": 1,
                    "event_name": "Test Grand Prix",
                }
            )
    return pd.DataFrame(rows)


def test_clean_frame_passes_every_gate():
    g = quality.run_gates(_laps())
    failed = g[~g["passed"]]["check"].tolist()
    assert not failed, f"clean data failed gates: {failed}"


def test_gate_catches_non_monotonic_lap_numbers():
    d = _laps()
    d.loc[5, "LapNumber"] = 2  # goes backwards
    g = quality.run_gates(d).set_index("check")
    assert not g.loc["lap_number_monotonic", "passed"]


def test_fresh_tyre_gate_passes_for_used_sets():
    """A scrubbed set legitimately starts older than the one just removed.

    An earlier version of this gate failed on exactly that, on 565 real
    stints. It was the gate that was wrong.
    """
    d = _laps()
    d["FreshTyre"] = False
    d.loc[d["Stint"] == 2, "TyreLife"] = d.loc[d["Stint"] == 2, "LapNumber"] + 30
    g = quality.run_gates(d).set_index("check")
    assert g.loc["fresh_tyre_starts_at_age_1", "passed"]


def test_fresh_tyre_gate_catches_a_fresh_set_starting_old():
    d = _laps()
    d["FreshTyre"] = True
    d.loc[d["Stint"] == 2, "TyreLife"] = d.loc[d["Stint"] == 2, "LapNumber"] + 30
    g = quality.run_gates(d).set_index("check")
    assert not g.loc["fresh_tyre_starts_at_age_1", "passed"]


def test_gate_catches_tyre_age_not_incrementing():
    d = _laps()
    d.loc[3, "TyreLife"] = 99
    g = quality.run_gates(d).set_index("check")
    assert not g.loc["tyre_age_increments", "passed"]


def test_gate_catches_impossible_lap_times():
    d = _laps()
    d.loc[2, "lap_time_s"] = 5.0  # no F1 lap is 5 seconds
    g = quality.run_gates(d).set_index("check")
    assert not g.loc["lap_time_in_bounds", "passed"]


def test_gate_catches_duplicate_car_laps():
    d = pd.concat([_laps(), _laps().head(1)], ignore_index=True)
    g = quality.run_gates(d).set_index("check")
    assert not g.loc["no_duplicate_car_laps", "passed"]


def test_gate_catches_unknown_circuit():
    d = _laps(circuit="atlantis")
    g = quality.run_gates(d).set_index("check")
    assert not g.loc["circuit_keys_known", "passed"]


def test_e1_excludes_a_race_with_almost_no_green_running():
    d = _laps(n_laps=8)
    d["is_green"] = True
    ex = quality.race_exclusions(d)
    assert bool(ex["E1_too_few_green_laps"].iloc[0])
    assert not bool(ex["use_for_strategy"].iloc[0])


def test_e5_excludes_a_wet_race():
    d = _laps()
    d.loc[d.index[: int(len(d) * 0.6)], "is_wet_tyre"] = True
    ex = quality.race_exclusions(d)
    assert bool(ex["E5_wet_race"].iloc[0])


def test_e3_excludes_an_early_red_flag():
    d = _laps(n_laps=40)
    d.loc[d["LapNumber"] <= 4, "TrackStatus"] = "5"
    ex = quality.race_exclusions(d)
    assert bool(ex["E3_early_red_flag"].iloc[0])


def test_excluded_races_still_feed_the_hazard_model():
    """The asymmetry the validation plan commits to.

    Removing the chaotic races from a model *of* chaos would bias it.
    """
    d = _laps(n_laps=40)
    d.loc[d["LapNumber"] <= 4, "TrackStatus"] = "5"
    ex = quality.race_exclusions(d)
    assert not bool(ex["use_for_strategy"].iloc[0])
    assert bool(ex["use_for_hazard"].iloc[0])


def test_green_lap_detection_requires_only_status_one():
    s = pd.Series(["1", "12", "1245", "", None, "4"])
    out = quality.pd.Series  # keep ruff quiet about unused import style
    from pitwall.dataset import is_green_lap

    g = is_green_lap(s)
    assert g.tolist() == [True, False, False, False, False, False]
    assert out is pd.Series


def test_sample_counts_reports_races_per_circuit_and_era():
    d = _laps()
    d["era"] = "2022-2025_ground_effect"
    counts = quality.sample_counts(d)
    assert counts["races"].iloc[0] == 1
    assert {"circuit", "era", "races", "laps"} <= set(counts.columns)


def test_filter_report_survives_missing_attrs():
    d = _laps()
    rep = quality.filter_report(d)
    assert "RETAINED" in rep["rule"].tolist()
    assert np.isfinite(rep["share_of_raw"]).all()
