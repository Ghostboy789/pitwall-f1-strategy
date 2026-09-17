"""Calibrate the stint-length cost term to what pit stops cost in real races.

Solves for one global coefficient on (tyre age)^2 such that the simulator
prices an extra pit stop the way real races did, fitting on one block of
seasons and reporting the moment in the held-out block. The optimiser is not
involved. See ``pitwall.models.degcurve`` for why this is a calibrated cost
term rather than a measured tyre cliff.

    python -m scripts.calibrate_deg_curve --fit-seasons 2018-2023

Writes ``models_out/deg_curvature.json``. Delete that file and the simulator
is linear in tyre age again, exactly as before.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from pitwall import config, pipeline, quality
from pitwall.models import degcurve

log = logging.getLogger("pitwall.calibrate_deg_curve")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate the degradation curvature.")
    ap.add_argument("--fit-seasons", default="2018-2023", help="seasons used to fit, inclusive")
    ap.add_argument("--races", type=int, default=40, help="races sampled for the simulated moment")
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--dry-run", action="store_true", help="report but do not write the artefact")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    lo, hi = (int(x) for x in args.fit_seasons.split("-"))
    art = pipeline.load_artifacts()
    laps = art["laps"]
    stints = pd.read_parquet(config.PROCESSED / "stints.parquet")

    excl = quality.race_exclusions(laps)
    usable = set(excl.loc[excl["use_for_strategy"], "race_id"])
    year_of = laps.groupby("race_id")["year"].first()
    circuit_of = laps.groupby("race_id")["circuit"].first()
    tp_circuits = set(art["trackposition"]["circuit"])

    fit_pool = sorted(
        r for r in usable if lo <= int(year_of[r]) <= hi and circuit_of.get(r) in tp_circuits
    )
    held_out = sorted(
        r for r in usable if int(year_of[r]) > hi and circuit_of.get(r) in tp_circuits
    )
    log.info(
        "fit pool %d races (%d-%d), held out %d races (%d+)",
        len(fit_pool),
        lo,
        hi,
        len(held_out),
        hi + 1,
    )

    # Spread the fit sample across circuits rather than taking a run of one season.
    rng = np.random.default_rng(args.seed)
    by_circuit: dict[str, list[str]] = {}
    for r in fit_pool:
        by_circuit.setdefault(circuit_of[r], []).append(r)
    picked: list[str] = []
    while len(picked) < min(args.races, len(fit_pool)):
        added = False
        for c in sorted(by_circuit):
            pool = [r for r in by_circuit[c] if r not in picked]
            if pool and len(picked) < args.races:
                picked.append(str(rng.choice(pool)))
                added = True
        if not added:
            break

    # The target moment, measured on the FIT seasons only.
    fit_laps = laps[laps["race_id"].isin(set(fit_pool))]
    fit_stints = stints[stints["race_id"].isin(set(fit_pool))]
    observed = degcurve.observed_cost_of_extra_stop(fit_laps, fit_stints)
    log.info(
        "observed cost of an extra stop, %d-%d: %+.2f s [%.2f, %.2f] over %d car-races in %d races",
        lo,
        hi,
        observed["beta_s_per_stop"],
        observed["ci_lo"],
        observed["ci_hi"],
        observed["n_car_races"],
        observed["n_races"],
    )

    result = degcurve.calibrate(
        fit_laps,
        fit_stints,
        art,
        picked,
        target=observed["beta_s_per_stop"],
        n_sims=args.n_sims,
    )

    log.info("=" * 66)
    log.info("solved curvature: %.6f s per lap per (lap of tyre age)^2", result["curvature"])
    if result.get("at_grid_edge"):
        log.warning("solution sits at the edge of the search grid - widen it")
    log.info("=" * 66)

    # What the calibrated curve implies, in units a reader can check.
    c = result["curvature"]
    for age in (5, 10, 20, 30, 40):
        log.info("  at tyre age %2d laps the curvature adds %5.2f s/lap", age, c * age * age)

    # The held-out moment, for the record. Not used to fit anything.
    ho_laps = laps[laps["race_id"].isin(set(held_out))]
    ho_stints = stints[stints["race_id"].isin(set(held_out))]
    held = degcurve.observed_cost_of_extra_stop(ho_laps, ho_stints)
    log.info(
        "held-out seasons %d+: observed cost of an extra stop %+.2f s [%.2f, %.2f] (n=%d)",
        hi + 1,
        held["beta_s_per_stop"],
        held["ci_lo"],
        held["ci_hi"],
        held["n_car_races"],
    )

    payload = {
        "curvature": result["curvature"],
        "units": "s per lap per (lap of tyre age)^2",
        "fit_seasons": args.fit_seasons,
        "n_fit_races_simulated": len(picked),
        "target_moment": observed,
        "held_out_moment": held,
        "grid": [[float(a), (float(b) if np.isfinite(b) else None)] for a, b in result["points"]],
        "status": result["status"],
        "method": (
            "Indirect inference. One global coefficient on (tyre age)^2, solved so "
            "the simulator reproduces what an extra pit stop actually cost teams, "
            "measured from finishing times with race fixed effects. Held-out lap "
            "times do not support the term, so it is a calibrated stint-length cost, "
            "not a measured tyre cliff. After this the sanity gate is no longer fully "
            "independent of the strategy layer and must not be reported as if it were."
        ),
    }
    if args.dry_run:
        log.info("dry run: not writing %s", config.MODELS_OUT / pipeline.DEG_CURVATURE_FILE)
        print(json.dumps(payload, indent=2, default=float))
        return 0

    path = config.MODELS_OUT / pipeline.DEG_CURVATURE_FILE
    path.write_text(json.dumps(payload, indent=2, default=float))
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
