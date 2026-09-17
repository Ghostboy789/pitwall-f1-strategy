"""Aggregates behind the website's Season and Overtaking dashboards.

    python -m scripts.export_web

Reads the Power BI extract in ``powerbi/data`` (committed, so this runs without
the raw lap data) and writes two small JSON files to ``models_out``, which is
what the deployed app reads. Both dashboards therefore show the same numbers as
the Power BI report. Aggregates are kept per race or per season, so the page can
re-total them for any season the visitor picks without a round trip.
"""

from __future__ import annotations

import json
import logging
import sys

import pandas as pd

from pitwall import config

log = logging.getLogger("pitwall.export_web")

DATA = config.ROOT / "powerbi" / "data"
GAP_BANDS = ["<0.5s", "0.5-1s", "1-1.5s", "1.5-2s", "2s+"]
PACE_BANDS = ["Slower", "0-0.25s", "0.25-0.5s", "0.5-1s", "1s+"]
RANKS = ["SOFTEST", "MIDDLE", "HARDEST", "WET"]


def _counts(frame: pd.DataFrame, by: list[str], name: str) -> pd.Series:
    return frame.groupby(by).size().rename(name)


def season_payload() -> dict:
    race = pd.read_csv(DATA / "dim_race.csv")
    circuit = pd.read_csv(DATA / "dim_circuit.csv").set_index("circuit_key")
    lap = pd.read_parquet(
        DATA / "fact_lap.parquet", columns=["race_id", "driver_code", "green_flag", "compound_rank"]
    )
    result = pd.read_csv(DATA / "fact_result.csv")
    stint = pd.read_csv(DATA / "fact_stint.csv")
    opp = pd.read_parquet(DATA / "fact_overtaking.parquet", columns=["race_id", "passed"])

    per_race = pd.concat(
        [
            _counts(lap, ["race_id"], "laps"),
            lap.groupby("race_id")["green_flag"].sum().rename("green_laps"),
            _counts(result, ["race_id"], "car_starts"),
            result.groupby("race_id")["classified"].sum().rename("classified"),
            stint.groupby("race_id")["ended_in_pit"].sum().rename("pit_stops"),
            opp.groupby("race_id")["passed"].sum().rename("passes"),
            _counts(opp, ["race_id"], "opportunities"),
            lap.groupby(["race_id", "compound_rank"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=RANKS, fill_value=0),
        ],
        axis=1,
    ).fillna(0)
    per_race = race.set_index("race_id")[["season", "circuit_key", "race_label"]].join(per_race)
    per_race["circuit"] = per_race["circuit_key"].map(circuit["circuit"])
    per_race["circuit_type"] = per_race["circuit_key"].map(circuit["circuit_type"])

    # Distinct counts cannot be re-totalled from per-race rows, so every
    # season-by-circuit-type combination is counted here.
    drivers = lap.merge(race[["race_id", "season", "circuit_key"]], on="race_id")
    drivers["circuit_type"] = drivers["circuit_key"].map(circuit["circuit_type"])
    driver_counts: dict[str, dict[str, int]] = {}
    for ctype, frame in (("all", drivers), *drivers.groupby("circuit_type")):
        counts = {"all": int(frame["driver_code"].nunique())}
        counts.update(
            {str(k): int(v) for k, v in frame.groupby("season")["driver_code"].nunique().items()}
        )
        driver_counts[ctype] = counts

    grid = result.dropna(subset=["grid_position", "positions_gained"]).merge(
        race[["race_id", "season"]], on="race_id"
    )
    grid = grid[grid["grid_position"].between(1, 20)]
    grid_rows = (
        grid.groupby(["season", "grid_position"])["positions_gained"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"grid_position": "grid", "sum": "gained", "count": "n"})
    )

    int_cols = [
        "laps",
        "green_laps",
        "car_starts",
        "classified",
        "pit_stops",
        "passes",
        "opportunities",
        *RANKS,
    ]
    per_race[int_cols] = per_race[int_cols].astype(int)
    return {
        "seasons": sorted(int(s) for s in race["season"].unique()),
        "drivers": driver_counts,
        "races": per_race.reset_index().to_dict("records"),
        "grid": [
            {"season": int(r.season), "grid": int(r.grid), "gained": float(r.gained), "n": int(r.n)}
            for r in grid_rows.itertuples()
        ],
        "ranks": RANKS,
    }


def overtaking_payload() -> dict:
    race = pd.read_csv(DATA / "dim_race.csv")[["race_id", "season", "circuit_key"]]
    circuit = pd.read_csv(DATA / "dim_circuit.csv").set_index("circuit_key")
    opp = pd.read_parquet(
        DATA / "fact_overtaking.parquet",
        columns=["race_id", "gap_band", "pace_band", "drs", "passed"],
    ).merge(race, on="race_id")
    metrics = pd.read_csv(DATA / "model_metrics.csv").iloc[0]
    calibration = pd.read_csv(DATA / "overtaking_reliability.csv")

    def cells(by: list[str]) -> list[dict]:
        g = opp.groupby(by)["passed"].agg(["sum", "count"]).reset_index()
        g = g.rename(columns={"sum": "passes", "count": "n"})
        return [
            {k: (int(v) if isinstance(v, (int, float)) and k != "drs" else v) for k, v in r.items()}
            for r in g.to_dict("records")
        ]

    by_circuit = cells(["season", "circuit_key"])
    for r in by_circuit:
        r["circuit"] = circuit.loc[r["circuit_key"], "circuit"]
    return {
        "seasons": sorted(int(s) for s in race["season"].unique()),
        "gap_bands": GAP_BANDS,
        "pace_bands": PACE_BANDS,
        "gap": [r for r in cells(["season", "gap_band", "drs"]) if r["gap_band"] in GAP_BANDS],
        "pace": [r for r in cells(["season", "pace_band"]) if r["pace_band"] in PACE_BANDS],
        "circuit": by_circuit,
        "calibration": calibration.rename(columns={"opportunities": "n"}).to_dict("records"),
        "model": {
            "auc": float(metrics["overtaking_auc"]),
            "brier": float(metrics["overtaking_brier"]),
            "ece": float(metrics["overtaking_ece"]),
        },
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    for name, payload in (
        ("web_season", season_payload()),
        ("web_overtaking", overtaking_payload()),
    ):
        path = config.MODELS_OUT / f"{name}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        log.info("%-16s %6.1f KB", name, path.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
