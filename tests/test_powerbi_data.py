"""The Power BI extract must agree with the fitted artefacts it was exported from.

These read the committed files in ``powerbi/data``, so they run in CI without
the raw lap data. They catch an export that drifts from ``models_out``.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pitwall import config

DATA = config.ROOT / "powerbi" / "data"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="Power BI extract not present")


def _metrics() -> dict:
    return json.loads((config.MODELS_OUT / "metrics.json").read_text())


def test_fact_tables_match_pipeline_counts() -> None:
    m = _metrics()
    laps = pd.read_parquet(DATA / "fact_lap.parquet")
    opps = pd.read_parquet(DATA / "fact_overtaking.parquet")
    stints = pd.read_csv(DATA / "fact_stint.csv")
    races = pd.read_csv(DATA / "dim_race.csv")

    assert len(races) == m["n_races"]
    assert len(laps) == m["n_laps_raw"]
    assert laps["driver_code"].nunique() == m["n_drivers"]
    assert len(stints) == m["n_stints"]
    assert len(opps) == m["n_opportunities"]
    assert opps["passed"].sum() == m["n_passes"]
    assert (~stints["tyre_life_observed"]).mean() == pytest.approx(m["share_stints_censored"])


def test_every_fact_row_joins_its_dimensions() -> None:
    races = set(pd.read_csv(DATA / "dim_race.csv")["race_id"])
    drivers = set(pd.read_csv(DATA / "dim_driver.csv")["driver_code"])
    circuits = set(pd.read_csv(DATA / "dim_circuit.csv")["circuit_key"])
    ranks = set(pd.read_csv(DATA / "dim_compound.csv")["compound_rank"])

    laps = pd.read_parquet(DATA / "fact_lap.parquet")
    opps = pd.read_parquet(DATA / "fact_overtaking.parquet")
    for name, frame in (
        ("laps", laps),
        ("opps", opps),
        ("results", pd.read_csv(DATA / "fact_result.csv")),
        ("stints", pd.read_csv(DATA / "fact_stint.csv")),
    ):
        assert set(frame["race_id"]) <= races, name
    assert set(laps["driver_code"]) <= drivers
    assert set(opps["follower_code"]) <= drivers
    assert set(laps["compound_rank"]) <= ranks
    assert set(pd.read_csv(DATA / "dim_race.csv")["circuit_key"]) <= circuits
    assert set(pd.read_csv(DATA / "degradation.csv")["circuit_key"]) <= circuits


def test_validation_evidence_matches_artefacts() -> None:
    gate = json.loads((config.MODELS_OUT / "sanity_gate.json").read_text())
    v6 = json.loads((config.MODELS_OUT / "strategy_validation.json").read_text())

    sanity = pd.read_csv(DATA / "sanity_gate.csv").set_index("metric")
    assert sanity.loc["Mean claimed gain (s)", "observed"] == pytest.approx(gate["mean_gain_s"])
    assert not sanity["passed"].any()

    backtest = pd.read_csv(DATA / "strategy_backtest.csv")
    row = backtest.query("source == 'Simulator − real' and strategy_cost == 'Extra pit stop'")
    assert row["estimate_s"].item() == pytest.approx(
        v6["simulated_minus_real"]["extra_stop_s"]["estimate"]
    )

    exclusions = pd.read_csv(config.MODELS_OUT / "race_exclusions.csv")
    races = pd.read_csv(DATA / "dim_race.csv")
    assert (~races["used_in_strategy_models"]).sum() == len(exclusions)


def test_withheld_audit_is_not_exported() -> None:
    names = {p.stem for p in DATA.iterdir()}
    assert not any("audit" in n for n in names)
