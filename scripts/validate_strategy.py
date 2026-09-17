"""Validation V6: compare strategy-level costs in real races and in the simulator.

    python -m scripts.validate_strategy

Writes ``models_out/strategy_validation.json``, which the dashboard reads.
See ``pitwall.models.strategy_validation`` for what is measured and why.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from pitwall import config, pipeline, quality
from pitwall.models import strategy_validation

log = logging.getLogger("pitwall.validate_strategy")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validation V6: strategy-level moments.")
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

    art = pipeline.load_artifacts()
    laps = art["laps"]
    stints = pd.read_parquet(config.PROCESSED / "stints.parquet")

    excl = quality.race_exclusions(laps)
    circuit_of = laps.groupby("race_id")["circuit"].first()
    tp = set(art["trackposition"]["circuit"])
    races = sorted(
        r for r in excl.loc[excl["use_for_strategy"], "race_id"] if circuit_of.get(r) in tp
    )

    out = strategy_validation.real_vs_simulated(
        laps, stints, art, races, n_sims=args.n_sims, n_boot=args.n_boot
    )
    path = config.MODELS_OUT / "strategy_validation.json"
    path.write_text(json.dumps(out, indent=2, default=float))

    for side in ("real", "simulated", "simulated_minus_real"):
        for k, v in out[side].items():
            log.info(
                "%-9s %-16s %+8.2f s  [%+.2f, %+.2f]",
                side,
                k,
                v["estimate"],
                v["ci_lo"],
                v["ci_hi"],
            )
    log.info("wrote %s (%d car-races, %d races)", path, out["n_car_races"], out["n_races"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
