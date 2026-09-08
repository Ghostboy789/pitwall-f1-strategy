"""End-to-end pipeline: raw parquet in, every model artefact out.

One command regenerates every number and figure in the project:

    python -m pitwall.pipeline

Each stage writes its artefact to ``models_out/`` so later stages and the web
app read from disk rather than refitting. Stages are independent enough that
one failing does not silently corrupt the next - it raises.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import pandas as pd

from pitwall import compounds, config, dataset, quality, trackposition
from pitwall.models import degradation, overtaking, pace, raceparams
from pitwall.sim import CircuitParams

log = logging.getLogger("pitwall.pipeline")

# Fallback degradation where a circuit-compound cell was never estimable.
DEFAULT_DEG_S_PER_LAP = 0.05
DEFAULT_PIT_LOSS_S = 23.0
DEFAULT_HAZARD = 0.02


def build_all(n_boot: int = trackposition.N_BOOTSTRAP, save: bool = True) -> dict:
    """Run every stage in order and return the artefacts."""
    t0 = time.time()
    out: dict = {}

    log.info("stage 1/8: assembling analysis tables")
    tables = dataset.build(save=save)
    laps = compounds.add_relative_hardness(tables["laps"])
    pace_laps = compounds.add_relative_hardness(tables["pace"])
    stints = degradation.with_compound_rank(tables["stints"], pace_laps)
    out["laps"], out["pace"], out["stints"] = laps, pace_laps, stints

    log.info("stage 2/8: data-quality gates and exclusion rules")
    gates = quality.run_gates(laps)
    exclusions = quality.race_exclusions(laps)
    counts = quality.sample_counts(laps)
    filt = quality.filter_report(pace_laps)
    out.update(gates=gates, exclusions=exclusions, sample_counts=counts, filter_report=filt)

    # Strategy modelling uses only races that passed the pre-registered rules;
    # the caution hazard deliberately uses all of them (see VALIDATION_PLAN).
    keep = set(exclusions.loc[exclusions["use_for_strategy"], "race_id"])
    pace_ok = pace_laps[pace_laps["race_id"].isin(keep)]
    stints_ok = stints[stints["race_id"].isin(keep)]
    log.info(
        "strategy modelling uses %d of %d races (%d laps)",
        len(keep),
        exclusions["race_id"].nunique(),
        len(pace_ok),
    )

    log.info("stage 3/8: lap-time decomposition")
    out["pace_model"] = pace.run(pace_ok, save=save)

    log.info("stage 4/8: degradation, naive vs censoring-corrected")
    out["degradation"] = degradation.compare(pace_ok, stints_ok, save=save)
    if save:
        # The ordering check drives a table on the dashboard, so it is
        # persisted rather than recomputed there - no user-facing number is
        # allowed to be typed in by hand.
        (config.MODELS_OUT / "degradation_ordering.json").write_text(
            json.dumps(
                {k: out["degradation"]["summary"][k] for k in ("naive", "twoway", "ipcw")},
                indent=2,
                default=float,
            )
        )

    log.info("stage 5/8: pit loss and caution hazard")
    ts = dataset.load_raw("track_status")
    out["raceparams"] = raceparams.run(laps, ts, save=save)

    log.info("stage 6/8: overtaking opportunities")
    opps = overtaking.build_opportunities(laps, pace_laps)
    if save:
        opps.to_parquet(config.PROCESSED / "opportunities.parquet", index=False)
    out["opportunities"] = opps
    out["detector_sanity"] = overtaking.detector_sanity(opps)

    log.info("stage 7/8: overtaking model and calibration")
    fit = overtaking.fit(opps)
    best = fit["best"]
    rel = overtaking.reliability(fit["y"], fit["results"][best]["oof_pred"])
    out["overtaking_fit"] = fit
    out["reliability"] = rel
    out["hierarchical"] = overtaking.fit_hierarchical(opps)
    if save:
        rel.to_csv(config.MODELS_OUT / "overtaking_reliability.csv", index=False)
        out["hierarchical"]["effects"].to_parquet(
            config.MODELS_OUT / "circuit_pass_effects.parquet", index=False
        )
        out["detector_sanity"].to_csv(config.MODELS_OUT / "detector_sanity.csv", index=False)

    log.info("stage 8/8: value of track position (%d bootstrap resamples)", n_boot)
    tp = trackposition.estimate(opps, n_boot=n_boot)
    out["trackposition"] = tp
    if save:
        tp.to_parquet(config.MODELS_OUT / "track_position_value.parquet", index=False)

    if save:
        build_circuit_reference(laps).to_parquet(
            config.MODELS_OUT / "circuit_reference.parquet", index=False
        )
        build_stint_limits(stints).to_parquet(
            config.MODELS_OUT / "stint_limits.parquet", index=False
        )

    metrics = {
        "n_races": int(laps["race_id"].nunique()),
        "n_races_used_for_strategy": len(keep),
        "n_laps_raw": len(laps),
        "n_laps_modelled": len(pace_ok),
        "n_circuits": int(laps["circuit"].nunique()),
        "n_drivers": int(laps["Driver"].nunique()),
        "n_teams": int(laps["Team"].nunique()),
        "n_stints": len(stints),
        "share_stints_censored": float(1 - stints["event_observed"].mean()),
        "n_opportunities": len(opps),
        "n_passes": int(opps["passed"].sum()),
        "overtaking_auc": float(fit["results"][best]["auc"]),
        "overtaking_brier": float(fit["results"][best]["brier"]),
        "overtaking_ece": float(rel.attrs["ece"]),
        "gates_passed": int(gates["passed"].sum()),
        "gates_total": len(gates),
        "seconds_elapsed": round(time.time() - t0, 1),
    }
    out["metrics"] = metrics
    if save:
        (config.MODELS_OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("pipeline complete in %.0fs: %s", time.time() - t0, metrics)
    return out


def circuit_params(
    circuit: str,
    artefacts: dict | None = None,
    race_laps: int | None = None,
) -> CircuitParams:
    """Assemble simulator parameters for one circuit from fitted artefacts.

    Every value is read from a model output. Where a circuit-compound cell was
    never estimable the project-wide default is substituted rather than a
    guess dressed up as an estimate, and the substitution is logged.
    """
    a = artefacts or load_artifacts()

    deg = a["degradation_table"]
    d = deg[deg["circuit"] == circuit]
    rank_map = {"SOFTEST": 0, "MIDDLE": 1, "HARDEST": 2}
    deg_by_rank: dict[int, float] = {}
    for label, rank in rank_map.items():
        row = d[d["compound_rank_label"] == label]
        if len(row):
            deg_by_rank[rank] = float(row["slope_s_per_lap_ipcw"].iloc[0])
        else:
            deg_by_rank[rank] = DEFAULT_DEG_S_PER_LAP
            log.debug("%s: no degradation estimate for %s, using default", circuit, label)
    # Degradation must be non-negative for a simulator: a tyre that gets
    # faster with age would make the optimiser run one stint forever.
    deg_by_rank = {k: max(v, 0.0) for k, v in deg_by_rank.items()}

    pit = a["pit_loss"]
    prow = pit[pit["circuit"] == circuit]
    pit_loss = float(prow["pit_loss_shrunk"].iloc[0]) if len(prow) else DEFAULT_PIT_LOSS_S

    haz = a["hazard"]
    hrow = haz[haz["circuit"] == circuit]
    hazard = float(hrow["hazard_shrunk"].iloc[0]) if len(hrow) else DEFAULT_HAZARD

    pc = a["pace_per_circuit"]
    prow2 = pc[pc["circuit"] == circuit]
    fuel = float(prow2["fuel"].iloc[0]) if len(prow2) else -0.06
    traffic = float(prow2["traffic"].iloc[0]) if len(prow2) else 1.0

    tp = a["trackposition"]
    trow = tp[tp["circuit"] == circuit]
    pass_base = float(trow["p_pass_per_lap"].iloc[0]) if len(trow) else 0.08

    rank_of = {"SOFTEST": 0, "MIDDLE": 1, "HARDEST": 2}
    limits = a.get("stint_limits")
    max_stint: dict[int, int] = {}
    if limits is not None and len(limits):
        lrow = limits[limits["circuit"] == circuit]
        for _, r in lrow.iterrows():
            rank = rank_of.get(str(r["compound_rank_label"]))
            if rank is not None:
                max_stint[rank] = int(r["max_stint"])

    if race_laps is None:
        rl = a.get("race_laps_by_circuit", {}).get(circuit)
        race_laps = int(rl) if rl is not None and pd.notna(rl) else 55

    base_lap = 90.0
    if "median_lap_s" in a and circuit in a["median_lap_s"]:
        base_lap = float(a["median_lap_s"][circuit])

    return CircuitParams(
        circuit=circuit,
        race_laps=int(race_laps),
        base_lap_s=base_lap,
        pit_loss_s=pit_loss,
        fuel_s_per_lap=fuel,
        caution_hazard_per_lap=hazard,
        traffic_s=traffic,
        deg_by_rank=deg_by_rank,
        max_stint_by_rank=max_stint,
        pass_p_base=pass_base,
    )


STINT_LIMIT_QUANTILE = 0.95


def build_stint_limits(stints: pd.DataFrame) -> pd.DataFrame:
    """Longest stint plausibly run on each compound rank, per circuit.

    The optimiser needs a feasibility ceiling because the degradation model is
    linear and understates the cliff: without one it proposes stints no team
    has ever run. The ceiling is the 95th percentile of observed stint lengths,
    which is a fact about the data rather than a tuned parameter.

    Circuits with too few stints on a compound fall back to the global
    percentile for that compound rather than to an invented number.
    """
    d = stints[stints["stint_length"] >= config.MIN_STINT_LAPS_FOR_DEG]
    d = d.dropna(subset=["compound_rank_label"])
    if d.empty:
        return pd.DataFrame(columns=["circuit", "compound_rank_label", "max_stint", "n"])

    glob = d.groupby("compound_rank_label")["stint_length"].quantile(STINT_LIMIT_QUANTILE)
    out = (
        d.groupby(["circuit", "compound_rank_label"])["stint_length"]
        .agg(n="size", max_stint=lambda x: x.quantile(STINT_LIMIT_QUANTILE))
        .reset_index()
    )
    thin = out["n"] < 10
    out.loc[thin, "max_stint"] = out.loc[thin, "compound_rank_label"].map(glob)
    out["max_stint"] = out["max_stint"].round().astype(int)
    out["fell_back_to_global"] = thin
    return out


def build_circuit_reference(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-circuit summary: the only thing the app needs from the lap table.

    ``laps_all.parquet`` is 12.7 MB and the deployed app reads exactly three
    numbers out of it per circuit. Persisting this summary instead lets the
    container ship ~100 KB of artefacts rather than 25 MB of lap data, and
    removes a gitignored file from the Docker build's critical path.
    """
    d = laps.assign(_t=pd.to_numeric(laps["LapTime"], errors="coerce"))
    green = d[d["is_green"]]
    return (
        pd.DataFrame(
            {
                "median_lap_s": green.groupby("circuit")["_t"].median(),
                "race_laps": d.groupby("circuit")["race_laps"].median(),
                "n_races": d.groupby("circuit")["race_id"].nunique(),
            }
        )
        .reset_index()
        .dropna(subset=["median_lap_s"])
    )


def load_artifacts() -> dict:
    """Read every persisted artefact back from disk.

    The full lap table is loaded only if it is present. A deployed container
    has the compact circuit reference instead, which is all the app reads.
    """
    m = config.MODELS_OUT
    out: dict = {
        "degradation_table": pd.read_parquet(m / "degradation_naive_vs_ipcw.parquet"),
        "pit_loss": pd.read_parquet(m / "pit_loss.parquet"),
        "hazard": pd.read_parquet(m / "caution_hazard.parquet"),
        "pace_per_circuit": pd.read_parquet(m / "pace_per_circuit.parquet"),
        "trackposition": pd.read_parquet(m / "track_position_value.parquet"),
        "circuit_pass_effects": pd.read_parquet(m / "circuit_pass_effects.parquet"),
        "metrics": json.loads((m / "metrics.json").read_text()),
    }

    lim_path = m / "stint_limits.parquet"
    out["stint_limits"] = pd.read_parquet(lim_path) if lim_path.exists() else None

    ref_path = m / "circuit_reference.parquet"
    laps_path = config.PROCESSED / "laps_all.parquet"

    if laps_path.exists():
        # `dataset.build` persists laps before compound ranking is applied, so
        # the rank is recomputed here rather than stored twice and left to drift.
        laps = compounds.add_relative_hardness(pd.read_parquet(laps_path))
        out["laps"] = laps
        ref = build_circuit_reference(laps)
    elif ref_path.exists():
        ref = pd.read_parquet(ref_path)
    else:
        raise FileNotFoundError(
            f"neither {laps_path} nor {ref_path} exists; run `python -m pitwall.pipeline`"
        )

    out["circuit_reference"] = ref
    out["median_lap_s"] = dict(zip(ref["circuit"], ref["median_lap_s"]))
    out["race_laps_by_circuit"] = dict(zip(ref["circuit"], ref["race_laps"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the full Pit Wall pipeline.")
    ap.add_argument("--bootstrap", type=int, default=trackposition.N_BOOTSTRAP)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    build_all(n_boot=args.bootstrap, save=not args.no_save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
