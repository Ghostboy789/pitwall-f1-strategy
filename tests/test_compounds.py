"""Trap T1: compound labels are relative. These tests pin that behaviour."""

from __future__ import annotations

import pandas as pd

from pitwall.compounds import add_relative_hardness, coverage_report, naming_scheme


def _laps(rows):
    return pd.DataFrame(rows, columns=["race_id", "Compound"])


def test_same_label_gets_different_rank_in_different_events():
    """The whole point of T1.

    At an event running SOFT/MEDIUM/HARD, SOFT is the softest available. At a
    2018 event running HYPERSOFT/ULTRASOFT/SOFT, the very same label is the
    *hardest* available. Pooling by label would merge two different tyres.
    """
    df = _laps(
        [("r1", "SOFT"), ("r1", "MEDIUM"), ("r1", "HARD")]
        + [("r2", "HYPERSOFT"), ("r2", "ULTRASOFT"), ("r2", "SOFT")]
    )
    out = add_relative_hardness(df)
    r1_soft = out[(out.race_id == "r1") & (out.Compound == "SOFT")]["compound_rank"].iloc[0]
    r2_soft = out[(out.race_id == "r2") & (out.Compound == "SOFT")]["compound_rank"].iloc[0]
    assert r1_soft == 0, "SOFT is the softest of SOFT/MEDIUM/HARD"
    assert r2_soft == 2, "SOFT is the hardest of HYPERSOFT/ULTRASOFT/SOFT"
    assert r1_soft != r2_soft


def test_rank_orders_softest_to_hardest_within_event():
    df = _laps([("r1", "HARD"), ("r1", "SOFT"), ("r1", "MEDIUM")])
    out = add_relative_hardness(df)
    ranks = out.set_index("Compound")["compound_rank"]
    assert ranks["SOFT"] < ranks["MEDIUM"] < ranks["HARD"]


def test_two_compound_event_ranks_zero_and_one():
    """Rank is relative to what was actually run, not to a fixed three."""
    df = _laps([("r1", "MEDIUM"), ("r1", "HARD")])
    out = add_relative_hardness(df)
    assert sorted(out["compound_rank"].unique().tolist()) == [0.0, 1.0]
    assert out["n_compounds_event"].iloc[0] == 2


def test_wet_tyres_excluded_from_dry_ranking():
    df = _laps([("r1", "SOFT"), ("r1", "MEDIUM"), ("r1", "INTERMEDIATE"), ("r1", "WET")])
    out = add_relative_hardness(df)
    wet = out[out["Compound"].isin(["INTERMEDIATE", "WET"])]
    assert wet["is_wet_tyre"].all()
    assert wet["compound_rank"].isna().all()
    assert out["n_compounds_event"].iloc[0] == 2


def test_compound_event_id_is_local_to_the_race():
    df = _laps([("r1", "SOFT"), ("r2", "SOFT")])
    out = add_relative_hardness(df)
    assert out["compound_event_id"].nunique() == 2


def test_coverage_report_counts_dry_compounds_per_race():
    df = pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r1", "r2", "r2"],
            "Compound": ["SOFT", "MEDIUM", "HARD", "SOFT", "MEDIUM"],
            "year": [2023, 2023, 2023, 2023, 2023],
        }
    )
    out = add_relative_hardness(df)
    cov = coverage_report(out)
    assert set(cov["n_dry_compounds"]) == {3, 2}


def test_naming_scheme_splits_at_2019():
    assert naming_scheme(2018) == "2018_seven_name"
    assert naming_scheme(2019) == "relative_smh"
