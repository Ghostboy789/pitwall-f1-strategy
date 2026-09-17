"""Run the counterfactual audit and the pre-registered sanity gate.

Separate from the pipeline because it is the expensive stage: every car in
every audited race is re-simulated against several alternative strategies.
Sampling is by race, controlled by ``--races``, so a quick pass and a thorough
one use the same code.

    python -m scripts.run_audit --races 40
    python -m scripts.run_audit --races 0 --shard 0/2   # every race, half of them
    python -m scripts.run_audit --races 0 --shard 1/2   # the other half, in parallel
    python -m scripts.run_audit --merge 2               # combine and apply the gate

Writes ``models_out/audit.parquet``, ``models_out/sanity_gate.json`` and the
team and driver summaries. The dashboard reads the gate verdict from disk and
says so honestly when it is absent.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

from pitwall import audit, config, pipeline

log = logging.getLogger("pitwall.run_audit")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Counterfactual audit + sanity gate.")
    ap.add_argument("--races", type=int, default=40, help="how many races to audit; 0 for all")
    ap.add_argument("--shard", default=None, help="k/n: audit only every n-th race, offset k")
    ap.add_argument("--merge", type=int, default=0, help="combine n finished shards, apply gate")
    ap.add_argument("--n-sims", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--max-stops", type=int, default=2)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--restart", action="store_true", help="discard partial results and start over")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    if args.merge:
        shards = [config.MODELS_OUT / f"audit_shard{k}.parquet" for k in range(args.merge)]
        missing = [p.name for p in shards if not p.exists()]
        if missing:
            log.error("cannot merge, shards not finished: %s", ", ".join(missing))
            return 1
        return _finish(pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True))

    art = pipeline.load_artifacts()
    laps = art["laps"]

    # Audit only races the pre-registered rules kept for strategy modelling.
    # A wet race or one killed by an early red flag has no strategic premise to
    # audit, and including it would put noise into the gate the gate is meant
    # to be measuring signal with.
    from pitwall import quality

    excl = quality.race_exclusions(laps)
    usable = excl.loc[excl["use_for_strategy"], "race_id"].tolist()

    circuit_of = laps.groupby("race_id")["circuit"].first()
    tp_circuits = set(art["trackposition"]["circuit"])
    usable = [r for r in usable if circuit_of.get(r) in tp_circuits]

    rng = np.random.default_rng(args.seed)
    if args.races and args.races < len(usable):
        # Spread the sample across circuits rather than taking a run of races
        # from one season.
        by_circuit: dict[str, list[str]] = {}
        for r in usable:
            by_circuit.setdefault(circuit_of[r], []).append(r)
        picked: list[str] = []
        while len(picked) < args.races:
            added = False
            for c in sorted(by_circuit):
                pool = [r for r in by_circuit[c] if r not in picked]
                if pool and len(picked) < args.races:
                    picked.append(str(rng.choice(pool)))
                    added = True
            if not added:
                break
        usable = picked

    tag = ""
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        usable = sorted(usable)[k::n]
        tag = f"_shard{k}"

    log.info("auditing %d races", len(usable))
    cfg = audit.AuditConfig(
        n_sims=args.n_sims, top_k=args.top_k, max_stops=args.max_stops, seed=args.seed
    )

    # Resumable, per race.
    #
    # A full pass is ~50 minutes and this has now been lost twice to the
    # session ending under it. Partial results are written after every race and
    # completed races are skipped on restart, so an interruption costs one race
    # rather than the run -- the same property ingestion has.
    partial_path = config.MODELS_OUT / f"audit_partial{tag}.parquet"
    frames: list[pd.DataFrame] = []
    done: set[str] = set()
    if partial_path.exists() and not args.restart:
        prev = pd.read_parquet(partial_path)
        if len(prev):
            frames.append(prev)
            done = set(prev["race_id"].unique())
            log.info("resuming: %d races already audited", len(done))

    params_cache: dict[str, object] = {}
    for i, race_id in enumerate(sorted(usable), 1):
        if race_id in done:
            continue
        circuit = circuit_of[race_id]
        if circuit not in params_cache:
            params_cache[circuit] = pipeline.circuit_params(circuit, art)
        t0 = time.time()
        try:
            res = audit.audit_race(laps, race_id, params_cache[circuit], cfg)
        except Exception as exc:
            log.warning("race %s failed: %s: %s", race_id, type(exc).__name__, exc)
            continue
        if res.empty:
            continue
        frames.append(res)
        pd.concat(frames, ignore_index=True).to_parquet(partial_path, index=False)
        log.info(
            "[%3d/%d] %-14s %s  %2d cars  mean gain %+.2fs  %4.1fs",
            i,
            len(usable),
            circuit,
            race_id,
            len(res),
            res["gain_s"].mean(),
            time.time() - t0,
        )

    if not frames:
        log.error("no races produced an audit")
        return 1

    a = pd.concat(frames, ignore_index=True)
    if tag:
        a.to_parquet(config.MODELS_OUT / f"audit{tag}.parquet", index=False)
        partial_path.unlink(missing_ok=True)
        log.info("shard done: %d car-races in %d races", len(a), a["race_id"].nunique())
        return 0
    partial_path.unlink(missing_ok=True)
    return _finish(a)


def _finish(a: pd.DataFrame) -> int:
    """Persist the audit, apply the pre-registered gate, and report it."""
    a.to_parquet(config.MODELS_OUT / "audit.parquet", index=False)
    gate = audit.sanity_gate(a)
    (config.MODELS_OUT / "sanity_gate.json").write_text(json.dumps(gate, indent=2, default=float))

    audit.team_summary(a).to_csv(config.MODELS_OUT / "audit_by_team.csv", index=False)
    audit.driver_summary(a).to_csv(config.MODELS_OUT / "audit_by_driver.csv", index=False)

    log.info("=" * 62)
    log.info("SANITY GATE: %s", "PASSED" if gate["passed"] else "FAILED")
    for k in (
        "n_car_races",
        "n_races",
        "mean_gain_s",
        "median_gain_s",
        "share_improved",
        "max_gain_s",
        "share_within_1s_of_optimal",
    ):
        log.info("  %-28s %s", k, gate[k])
    for f in gate["failures"]:
        log.info("  FAILURE: %s", f)
    log.info("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
