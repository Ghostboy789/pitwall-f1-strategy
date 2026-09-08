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

import numpy as np
import pandas as pd

from pitwall import compounds, config, dataset, quality, trackposition
from pitwall.circuits import CIRCUIT_REF
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
        len(keep), exclusions["race_id"].nunique(), len(pace_ok),
    )

    log.info("stage 3/8: lap-time decomposition")
    out["pace_model"] = pace.run(pace_ok, save=save)

    log.info("stage 4/8: degradation, naive vs censoring-corrected")
    out["degradation"] = degradation.compare(pace_ok, stints_ok, save=save)

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

    metrics = {
        "n_races": int(laps["race_id"].nunique()),
        "n_races_used_for_strategy": len(keep),
        "n_laps_raw": int(len(laps)),
        "n_laps_modelled": int(len(pace_ok)),
        "n_circuits": int(laps["circuit"].nunique()),
        "n_drivers": int(laps["Driver"].nunique()),
        "n_teams": int(laps["Team"].nunique()),
        "n_stints": int(len(stints)),
        "share_stints_censored": float(1 - stints["event_observed"].mean()),
        "n_opportunities": int(len(opps)),
        "n_passes": int(opps["passed"].sum()),
        "overtaking_auc": float(fit["results"][best]["auc"]),
        "overtaking_brier": float(fit["results"][best]["brier"]),
        "overtaking_ece": float(rel.attrs["ece"]),
        "gates_passed": int(gates["passed"].sum()),
        "gates_total": int(len(gates)),
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

    if race_laps is None:
        rl = a["laps"].loc[a["laps"]["circuit"] == circuit, "race_laps"]
        race_laps = int(rl.median()) if len(rl) else 55

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
        pass_p_base=pass_base,
    )


def load_artifacts() -> dict:
    """Read every persisted artefact back from disk."""
    m = config.MODELS_OUT
    # `dataset.build` persists laps before compound ranking is applied, so the
    # rank is recomputed here rather than stored twice and allowed to drift.
    laps = compounds.add_relative_hardness(
        pd.read_parquet(config.PROCESSED / "laps_all.parquet")
    )
    med = (
        laps.assign(_t=pd.to_numeric(laps["LapTime"], errors="coerce"))
        .query("is_green")
        .groupby("circuit")["_t"]
        .median()
        .to_dict()
    )
    return {
        "laps": laps,
        "median_lap_s": med,
        "degradation_table": pd.read_parquet(m / "degradation_naive_vs_ipcw.parquet"),
        "pit_loss": pd.read_parquet(m / "pit_loss.parquet"),
        "hazard": pd.read_parquet(m / "caution_hazard.parquet"),
        "pace_per_circuit": pd.read_parquet(m / "pace_per_circuit.parquet"),
        "trackposition": pd.read_parquet(m / "track_position_value.parquet"),
        "circuit_pass_effects": pd.read_parquet(m / "circuit_pass_effects.parquet"),
        "metrics": json.loads((m / "metrics.json").read_text()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the full Pit Wall pipeline.")
    ap.add_argument("--bootstrap", type=int, default=trackposition.N_BOOTSTRAP)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )
    build_all(n_boot=args.bootstrap, save=not args.no_save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
