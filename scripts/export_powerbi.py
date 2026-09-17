"""Export the tables behind the Power BI report in ``powerbi/``.

    python -m scripts.export_powerbi

Reads the processed lap data and the fitted artefacts, and writes a small star
schema to ``powerbi/data``: dimensions for race, circuit and driver; facts for
laps, results, stints and overtaking opportunities; the validation evidence; and
the team colour themes, which are also written to ``models_out`` for the website.
The report loads these files straight from GitHub, so a clone opens without the
~50 MB ingestion. Nothing here fits a model: every number is read from
``models_out`` or recounted from the lap table.

The per-team and per-driver strategy audit is not exported. It failed the
pre-registered sanity gate and is withheld everywhere, including here.
"""

from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pandas as pd

from pitwall import config
from pitwall.circuits import CIRCUIT_REF
from pitwall.team_colours import team_themes

log = logging.getLogger("pitwall.export_powerbi")

OUT = config.ROOT / "powerbi" / "data"

GATE_LABELS = {
    "non_empty": "Lap table has rows",
    "lap_number_monotonic": "Lap numbers rise per car",
    "stint_non_decreasing": "Stints never go backwards",
    "tyre_age_increments": "Tyre age +1 per lap in a stint",
    "fresh_tyre_starts_at_age_1": "New tyres start at age 1",
    "lap_time_in_bounds": "Lap time within 50–400 s",
    "circuit_keys_known": "Every circuit is a known layout",
    "no_duplicate_car_laps": "One row per car per lap",
}
EXCLUSION_LABELS = {
    "E1_too_few_green_laps": "Too few green-flag laps",
    "E3_early_red_flag": "Red flag in the first quarter",
    "E5_wet_race": "Wet race",
}
COMPOUND_ORDER = {"SOFTEST": 1, "MIDDLE": 2, "HARDEST": 3, "WET": 4, "UNKNOWN": 5}


# Chart-axis labels: full layout names truncate on a 28-circuit axis.
SHORT_NAMES = {
    "albert_park": "Melbourne",
    "bahrain": "Bahrain",
    "bahrain_outer": "Sakhir Outer",
    "baku": "Baku",
    "catalunya": "Barcelona",
    "cota": "Austin",
    "hockenheim": "Hockenheim",
    "hungaroring": "Budapest",
    "imola": "Imola",
    "interlagos": "São Paulo",
    "istanbul": "Istanbul",
    "jeddah": "Jeddah",
    "las_vegas": "Las Vegas",
    "lusail": "Lusail",
    "singapore": "Singapore",
    "rodriguez": "Mexico City",
    "miami": "Miami",
    "monaco": "Monaco",
    "villeneuve": "Montréal",
    "monza": "Monza",
    "mugello": "Mugello",
    "nurburgring": "Nürburgring",
    "paul_ricard": "Le Castellet",
    "portimao": "Portimão",
    "red_bull_ring": "Spielberg",
    "shanghai": "Shanghai",
    "silverstone": "Silverstone",
    "sochi": "Sochi",
    "spa": "Spa",
    "suzuka": "Suzuka",
    "yas_marina": "Abu Dhabi",
    "zandvoort": "Zandvoort",
}


def _name(key: str) -> str:
    ref = CIRCUIT_REF.get(key)
    return ref.name if ref else key.replace("_", " ").title()


def _compound(label: pd.Series, wet: pd.Series) -> pd.Series:
    label = label.astype(str).str.upper()
    known = label.map(
        lambda c: c in {"HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"}
    )
    return label.where(known, "UNKNOWN").where(~wet, "WET")


def _band(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(values, edges, labels=labels, right=False).astype(str).replace("nan", "No data")


def _band_order(band: pd.Series, labels: list[str]) -> pd.Series:
    """Sort key for a band label, so Power BI orders bands by value, not alphabet."""
    return (
        band.map({b: i for i, b in enumerate(labels, start=1)}).fillna(len(labels) + 1).astype(int)
    )


def races(laps: pd.DataFrame, exclusions: pd.DataFrame) -> pd.DataFrame:
    g = laps.groupby("race_id")
    out = pd.DataFrame(
        {
            "season": g["year"].first(),
            "round": g["round"].first(),
            "event_name": g["event_name"].first(),
            "circuit_key": g["circuit"].first(),
            "event_date": pd.to_datetime(g["event_date"].first()).dt.date,
            "race_laps": g["race_laps"].max().astype(int),
            "cars": g["car_id"].nunique(),
            "laps_recorded": g.size(),
            "green_lap_share": g["is_green"].mean().round(4),
        }
    ).reset_index()
    ex = exclusions.set_index("race_id")
    out["exclusion_reason"] = (
        out["race_id"]
        .map(ex["exclusion_reason"])
        .map(lambda r: " + ".join(EXCLUSION_LABELS[c] for c in r.split("|")), na_action="ignore")
    )
    out["used_in_strategy_models"] = out["exclusion_reason"].isna()
    out["race_label"] = out["season"].astype(str) + " " + out["event_name"]
    return out


def circuits(laps: pd.DataFrame, opps: pd.DataFrame) -> pd.DataFrame:
    keys = sorted(laps["circuit"].unique())
    ref = pd.DataFrame(
        {
            "circuit_key": keys,
            "circuit": [_name(k) for k in keys],
            "short_name": [SHORT_NAMES.get(k, _name(k)) for k in keys],
            "lap_km": [getattr(CIRCUIT_REF.get(k), "lap_km", np.nan) for k in keys],
            "drs_zones": [getattr(CIRCUIT_REF.get(k), "drs_zones", np.nan) for k in keys],
            "circuit_type": [
                "Street" if getattr(CIRCUIT_REF.get(k), "street", False) else "Permanent"
                for k in keys
            ],
        }
    )
    tp = pd.read_parquet(config.MODELS_OUT / "track_position_value.parquet").rename(
        columns={"circuit": "circuit_key"}
    )
    tp = tp[
        [
            "circuit_key",
            "p_pass_per_lap",
            "p_lo",
            "p_hi",
            "value_s",
            "value_lo",
            "value_hi",
            "at_ceiling",
        ]
    ].rename(
        columns={
            "p_pass_per_lap": "pass_chance_per_lap",
            "p_lo": "pass_chance_lo",
            "p_hi": "pass_chance_hi",
            "value_s": "cost_of_being_stuck_s",
            "value_lo": "cost_lo_s",
            "value_hi": "cost_hi_s",
        }
    )
    pit = pd.read_parquet(config.MODELS_OUT / "pit_loss.parquet").rename(
        columns={"circuit": "circuit_key"}
    )[["circuit_key", "pit_loss_shrunk", "pit_loss_shrunk_se"]]
    pit.columns = ["circuit_key", "pit_loss_s", "pit_loss_se_s"]
    caution = pd.read_parquet(config.MODELS_OUT / "caution_hazard.parquet").rename(
        columns={"circuit": "circuit_key"}
    )[["circuit_key", "expected_cautions_per_race"]]
    passes = (
        opps.groupby("circuit")
        .agg(opportunities=("passed", "size"), passes=("passed", "sum"))
        .reset_index()
        .rename(columns={"circuit": "circuit_key"})
    )
    out = ref
    for t in (tp, pit, caution, passes):
        out = out.merge(t, on="circuit_key", how="left")
    return out


def drivers(laps: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Every driver code that appears in a lap or a result row."""
    results = results.sort_values(["year", "round"])
    seen = pd.concat(
        [
            laps[["year", "round", "Driver", "Team"]],
            results[["year", "round", "Abbreviation", "TeamName"]].set_axis(
                ["year", "round", "Driver", "Team"], axis=1
            ),
        ]
    ).dropna(subset=["Driver"])
    seen = seen[seen["Driver"].astype(str).str.len() == 3].sort_values(["year", "round"])
    g = seen.groupby("Driver")
    out = pd.DataFrame(
        {
            "latest_team": g["Team"].last(),
            "first_season": g["year"].min(),
            "last_season": g["year"].max(),
        }
    ).reset_index(names="driver_code")
    names = results.groupby("Abbreviation")["FullName"].last()
    out["driver"] = out["driver_code"].map(names).fillna(out["driver_code"])
    return out[["driver_code", "driver", "latest_team", "first_season", "last_season"]]


def _race_id(df: pd.DataFrame) -> pd.Series:
    return df["year"].astype(str) + "_" + df["round"].astype(int).map("{:02d}".format)


def results_table(
    results: pd.DataFrame, classification: pd.DataFrame, race_ids: set[str]
) -> pd.DataFrame:
    """FastF1 results, with grid/status/points filled from the Ergast backfill.

    From 2022 FastF1 stopped returning those fields for most races;
    ``pitwall.backfill_results`` recovered them separately.
    """
    r = results.assign(race_id=_race_id(results))
    r = r[r["race_id"].isin(race_ids)].reset_index(drop=True)
    c = classification.assign(race_id=_race_id(classification)).set_index(["race_id", "driverCode"])
    key = pd.MultiIndex.from_arrays([r["race_id"], r["Abbreviation"]])
    backfill = c.reindex(key).reset_index(drop=True)
    r["GridPosition"] = pd.to_numeric(r["GridPosition"], errors="coerce").fillna(
        pd.to_numeric(backfill["grid"], errors="coerce")
    )
    r["Status"] = r["Status"].replace("", np.nan).fillna(backfill["status"])
    r["Points"] = pd.to_numeric(r["Points"], errors="coerce").fillna(
        pd.to_numeric(backfill["points"], errors="coerce")
    )
    text = r["ClassifiedPosition"].where(
        r["ClassifiedPosition"].astype(str).str.strip().ne(""), backfill["positionText"]
    )
    finish = pd.to_numeric(r["Position"], errors="coerce")
    grid = r["GridPosition"].replace(0, np.nan)
    classified = text.astype(str).str.isdigit()
    return pd.DataFrame(
        {
            "race_id": r["race_id"],
            "driver_code": r["Abbreviation"],
            "team": r["TeamName"],
            "grid_position": grid,
            "finish_position": finish,
            "classified": classified,
            "status": r["Status"],
            "points": pd.to_numeric(r["Points"], errors="coerce"),
            "positions_gained": np.where(classified, grid - finish, np.nan),
        }
    ).reset_index(drop=True)


def laps_table(laps: pd.DataFrame, pace: pd.DataFrame) -> pd.DataFrame:
    pace_key = set(zip(pace["car_id"], pace["LapNumber"]))
    is_pace = pd.Series(
        [k in pace_key for k in zip(laps["car_id"], laps["LapNumber"])], index=laps.index
    )
    lap_time = laps["LapTime"].where(laps["LapTime"].between(1, 1000))
    median = lap_time.where(is_pace).groupby(laps["race_id"]).transform("median")
    compound = _compound(laps["Compound"], laps["is_wet_tyre"])
    label = pd.Series(np.nan, index=laps.index, dtype=object)
    if "compound_rank_label" in laps:
        label = laps["compound_rank_label"]
    out = pd.DataFrame(
        {
            "race_id": laps["race_id"],
            "driver_code": laps["Driver"],
            "team": laps["Team"],
            "lap": laps["LapNumber"].astype(int),
            "lap_time_s": lap_time.round(3),
            "compound": compound,
            "compound_rank": label.where(~laps["is_wet_tyre"], "WET").fillna("UNKNOWN"),
            "tyre_age_laps": laps["TyreLife"],
            "stint": laps["Stint"],
            "position": laps["Position"],
            "green_flag": laps["is_green"],
            "pit_in_lap": laps["is_inlap"],
            "pit_out_lap": laps["is_outlap"],
            "pace_lap": is_pace,
            "delta_to_race_median_s": (lap_time - median).where(is_pace).round(3),
        }
    )
    return out


def stints_table(stints: pd.DataFrame, rank: pd.DataFrame) -> pd.DataFrame:
    s = stints.merge(rank, on=["race_id", "compound"], how="left")
    wet = s["compound"].isin(config.WET_LABELS)
    compound = _compound(s["compound"], wet)
    return pd.DataFrame(
        {
            "stint_id": s["stint_id"],
            "race_id": s["race_id"],
            "driver_code": s["driver"],
            "team": s["team"],
            "stint": s["stint"].astype("Int64"),
            "compound": compound,
            "compound_rank": s["compound_rank"].where(~wet, "WET").fillna("UNKNOWN"),
            "fresh_tyre": s["fresh_tyre"],
            "start_lap": s["start_lap"],
            "end_lap": s["end_lap"],
            "stint_laps": s["stint_length"],
            "ended_in_pit": s["ended_in_pit"],
            "ran_to_flag": s["reached_race_end"],
            "tyre_life_observed": s["event_observed"],
            "caution_during_stint": s["any_non_green"],
        }
    )


GAP_BANDS = ["<0.5s", "0.5-1s", "1-1.5s", "1.5-2s", "2s+"]
PACE_BANDS = ["Slower", "0-0.25s", "0.25-0.5s", "0.5-1s", "1s+"]


def opportunities_table(opps: pd.DataFrame) -> pd.DataFrame:
    advantage = -opps["pace_delta_s"]
    gap_band = _band(opps["gap_s"], [0, 0.5, 1.0, 1.5, 2.0, 99], GAP_BANDS)
    pace_band = _band(advantage, [-99, 0, 0.25, 0.5, 1.0, 99], PACE_BANDS)
    return pd.DataFrame(
        {
            "race_id": opps["race_id"],
            "lap": opps["lap"],
            "follower_code": opps["follower"].str[-3:],
            "leader_code": opps["leader"].str[-3:],
            "position": opps["position"],
            "gap_s": opps["gap_s"].round(3),
            "gap_band": gap_band,
            "gap_band_order": _band_order(gap_band, GAP_BANDS),
            "pace_advantage_s": advantage.round(3),
            "pace_band": pace_band,
            "pace_band_order": _band_order(pace_band, PACE_BANDS),
            "drs": np.where(opps["drs_available"] == 1, "DRS open", "No DRS"),
            "passed": opps["passed"].astype(int),
            "race_progress": opps["lap_progress"].round(3),
        }
    )


def degradation_table() -> pd.DataFrame:
    d = pd.read_parquet(config.MODELS_OUT / "degradation_naive_vs_ipcw.parquet")
    rows = []
    for est, slope, se in (
        ("1 Naive", "slope_naive", "se_naive"),
        ("2 Fixed effects", "slope_twoway", "se_twoway"),
        ("3 IPCW-corrected", "slope_ipcw", "se_ipcw_race_clustered"),
        ("4 Pooled (simulator)", "slope_s_per_lap_sim", "se_ipcw_race_clustered"),
    ):
        rows.append(
            pd.DataFrame(
                {
                    "circuit_key": d["circuit"],
                    "compound_rank": d["compound_rank_label"],
                    "estimator": est[2:],
                    "estimator_order": int(est[0]),
                    "wear_s_per_lap": d[slope],
                    "se_s_per_lap": d[se],
                    "laps": d["n_laps"],
                    "stints": d["n_stints"],
                    "races": d["n_races"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def validation_tables() -> dict[str, pd.DataFrame]:
    metrics = json.loads((config.MODELS_OUT / "metrics.json").read_text())
    gate = json.loads((config.MODELS_OUT / "sanity_gate.json").read_text())
    v6 = json.loads((config.MODELS_OUT / "strategy_validation.json").read_text())
    ordering = json.loads((config.MODELS_OUT / "degradation_ordering.json").read_text())
    q = pd.read_csv(config.MODELS_OUT / "data_quality_gates.csv")

    checks = pd.DataFrame(
        [
            (
                "V1",
                "Predict finishing order, held-out seasons",
                "Not run",
                "Listed in the plan, not dropped",
            ),
            (
                "V2",
                "Overtaking model calibrated (out-of-fold)",
                "Pass",
                f"AUC {metrics['overtaking_auc']:.3f} · Brier {metrics['overtaking_brier']:.4f} · "
                f"ECE {metrics['overtaking_ece']:.4f} on {metrics['n_opportunities']:,} opportunities",
            ),
            (
                "V3",
                "Optimiser gain stays plausible",
                "Fail",
                f"Mean claimed gain {gate['mean_gain_s']:.1f} s against a "
                f"{gate['thresholds']['max_mean_gain_s']:.1f} s limit",
            ),
            (
                "V4",
                "Ablations",
                "Partial",
                "Leakage only: removing the pace feature costs 0.015 AUC",
            ),
            (
                "V5",
                "Structural data-quality gates",
                "Pass",
                f"{int(q['passed'].sum())} of {len(q)} across {metrics['n_laps_raw']:,} laps",
            ),
            (
                "V6",
                "Simulator prices strategy like real races",
                "Fail",
                f"Extra stop costs +{v6['simulated_minus_real']['extra_stop_s']['estimate']:.1f} s "
                "more in simulation than in real races",
            ),
        ],
        columns=["check_id", "check", "status", "result"],
    )
    checks["status_order"] = checks["status"].map(
        {"Fail": 1, "Partial": 2, "Not run": 3, "Pass": 4}
    )

    t = gate["thresholds"]
    sanity = pd.DataFrame(
        [
            ("Mean claimed gain (s)", gate["mean_gain_s"], t["max_mean_gain_s"]),
            ("Share of car-races 'improved'", gate["share_improved"], t["max_share_improved"]),
            ("Largest single claimed gain (s)", gate["max_gain_s"], t["max_single_gain_s"]),
        ],
        columns=["metric", "observed", "limit"],
    )
    sanity["passed"] = sanity["observed"] <= sanity["limit"]
    sanity["car_races"] = gate["n_car_races"]
    sanity["races"] = gate["n_races"]

    labels = {"extra_stop_s": "Extra pit stop", "stint_balance_s": "Uneven stints"}
    sources = {
        "real": "Real races",
        "simulated": "Simulator",
        "simulated_minus_real": "Simulator − real",
    }
    backtest = pd.DataFrame(
        [
            (
                labels[k],
                sources[s],
                v6[s][k]["estimate"],
                v6[s][k]["ci_lo"],
                v6[s][k]["ci_hi"],
                v6[s][k]["n_races"],
            )
            for s in sources
            for k in labels
        ],
        columns=["strategy_cost", "source", "estimate_s", "ci_lo_s", "ci_hi_s", "races"],
    )

    # The quality module states each gate's tolerance in its detail text.
    allowed = q["detail"].str.extract(r"tolerance (\d+)")[0].fillna(0).astype(int)
    gates = q.assign(check=q["check"].map(GATE_LABELS).fillna(q["check"]), allowed=allowed)[
        ["check", "n_checked", "n_violations", "allowed", "passed"]
    ]
    rel = pd.read_csv(config.MODELS_OUT / "overtaking_reliability.csv")
    rel = rel.rename(columns={"bin": "decile", "n": "opportunities"})[
        ["decile", "opportunities", "mean_predicted", "observed_rate"]
    ]
    order = pd.DataFrame(
        [
            (
                name,
                i,
                ordering[k]["n_softest_faster_than_hardest"],
                ordering[k]["n_circuits"],
                ordering[k]["n_negative_cells"],
                ordering[k]["n_cells"],
            )
            for i, (k, name) in enumerate(
                [
                    ("naive", "Naive"),
                    ("twoway", "Fixed effects"),
                    ("ipcw", "IPCW-corrected"),
                ],
                start=1,
            )
        ],
        columns=[
            "estimator",
            "estimator_order",
            "circuits_soft_wears_fastest",
            "circuits",
            "impossible_cells",
            "cells",
        ],
    )
    kpis = pd.DataFrame([{k: v for k, v in metrics.items() if k != "seconds_elapsed"}])
    return {
        "validation_checks": checks,
        "sanity_gate": sanity,
        "strategy_backtest": backtest,
        "quality_gates": gates,
        "overtaking_reliability": rel,
        "tyre_estimator_ordering": order,
        "model_metrics": kpis,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    from pitwall.compounds import add_relative_hardness

    OUT.mkdir(parents=True, exist_ok=True)
    laps = add_relative_hardness(pd.read_parquet(config.PROCESSED / "laps_all.parquet"))
    pace = pd.read_parquet(config.PROCESSED / "laps_pace.parquet", columns=["car_id", "LapNumber"])
    opps = pd.read_parquet(config.PROCESSED / "opportunities.parquet")
    stints = pd.read_parquet(config.PROCESSED / "stints.parquet")
    results = pd.concat(
        [pd.read_parquet(p) for p in sorted((config.RAW / "results").glob("*.parquet"))],
        ignore_index=True,
    )
    classification = pd.concat(
        [pd.read_parquet(p) for p in sorted((config.RAW / "classification").glob("*.parquet"))],
        ignore_index=True,
    )
    exclusions = pd.read_csv(config.MODELS_OUT / "race_exclusions.csv")

    rank = (
        laps.dropna(subset=["compound_rank_label"])[["race_id", "Compound", "compound_rank_label"]]
        .drop_duplicates()
        .rename(columns={"Compound": "compound", "compound_rank_label": "compound_rank"})
    )
    race = races(laps, exclusions)
    tables: dict[str, pd.DataFrame] = {
        "dim_race": race,
        "dim_circuit": circuits(laps, opps),
        "dim_driver": drivers(laps, results),
        "dim_compound": pd.DataFrame(
            {"compound_rank": list(COMPOUND_ORDER), "compound_order": list(COMPOUND_ORDER.values())}
        ),
        "fact_result": results_table(results, classification, set(race["race_id"])),
        "fact_stint": stints_table(stints, rank),
        "degradation": degradation_table(),
        "team_themes": team_themes(results),
        **validation_tables(),
    }
    big = {"fact_lap": laps_table(laps, pace), "fact_overtaking": opportunities_table(opps)}

    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        log.info("%-24s %7d rows", name, len(df))
    for name, df in big.items():
        df.to_parquet(OUT / f"{name}.parquet", index=False, compression="zstd")
        log.info("%-24s %7d rows", name, len(df))
    # The website reads the same themes from the deployed artefacts.
    tables["team_themes"].to_csv(config.MODELS_OUT / "team_themes.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
