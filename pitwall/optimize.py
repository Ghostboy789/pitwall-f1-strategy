"""Phase 6 - strategy optimisation.

Two stages, for a reason that is about honesty as much as speed.

**Stage 1, closed form.** Ignoring traffic and cautions, a strategy's race time
has an exact expression. For a stint of ``n`` laps on a compound degrading at
``d`` seconds per lap, the tyre costs ``d * n(n+1)/2`` over the stint, and
everything else - base pace, fuel burn - is identical across strategies with
the same number of stops. So the ranking reduces to

    minimise   sum_stints d_i * n_i(n_i+1)/2  +  n_stops * pit_loss

which is enumerable over every stop lap and compound sequence in milliseconds.

**Stage 2, Monte Carlo.** The closed form deliberately cannot see the two
things that actually decide races: traffic and safety cars. So the top
candidates from stage 1 are re-run through the full simulator, which can. The
ordering frequently changes, and where it does, that difference *is* the
finding - it is the value of track position expressing itself.

No reinforcement learning, no genetic algorithm. The space is small enough to
enumerate exhaustively, and an exhaustive search that can be checked by hand
is worth more here than a clever one that cannot.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitwall import config
from pitwall.sim import Car, CircuitParams, Strategy, simulate

log = logging.getLogger("pitwall.optimize")

# A stint shorter than this is not a strategy, it is a mistake.
MIN_STINT_LAPS = 5
MAX_STOPS = 3
TOP_K_FOR_SIMULATION = 12


@dataclass(frozen=True)
class Candidate:
    """One enumerated strategy and its closed-form tyre-plus-pit cost."""

    strategy: Strategy
    deterministic_cost_s: float
    n_stops: int
    stint_lengths: tuple[int, ...]


def deterministic_cost(
    stop_laps: tuple[int, ...],
    ranks: tuple[int, ...],
    race_laps: int,
    deg_by_rank: dict[int, float],
    pit_loss_s: float,
) -> tuple[float, tuple[int, ...]]:
    """Tyre cost plus pit cost for one strategy. Lower is better.

    Only the terms that differ between strategies are included: base pace and
    fuel burn are the same for every strategy over the same race distance, so
    they cancel out of the comparison and are left out rather than carried
    around as a constant.
    """
    boundaries = (0, *stop_laps, race_laps)
    lengths = tuple(boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1))
    if any(n < MIN_STINT_LAPS for n in lengths):
        return np.inf, lengths

    tyre = sum(deg_by_rank.get(rank, 0.05) * n * (n + 1) / 2.0 for n, rank in zip(lengths, ranks))
    return tyre + len(stop_laps) * pit_loss_s, lengths


def enumerate_strategies(
    params: CircuitParams,
    max_stops: int = MAX_STOPS,
    available_ranks: tuple[int, ...] = (0, 1, 2),
    require_two_compounds: bool = True,
) -> list[Candidate]:
    """Every legal strategy, ranked by the closed-form cost.

    ``require_two_compounds`` enforces the regulation that a dry race must use
    at least two different compounds. It is a rule of the sport, not a
    modelling choice, and dropping it would let the optimiser "win" by
    proposing something that would be disqualified.
    """
    out: list[Candidate] = []
    laps = params.race_laps

    for n_stops in range(1, max_stops + 1):
        for stop_laps in itertools.combinations(
            range(MIN_STINT_LAPS, laps - MIN_STINT_LAPS + 1), n_stops
        ):
            for ranks in itertools.product(available_ranks, repeat=n_stops + 1):
                if require_two_compounds and len(set(ranks)) < 2:
                    continue
                cost, lengths = deterministic_cost(
                    stop_laps, ranks, laps, params.deg_by_rank, params.pit_loss_s
                )
                if not np.isfinite(cost):
                    continue
                out.append(
                    Candidate(
                        strategy=Strategy(
                            stops=tuple(zip(stop_laps, ranks[1:])), start_rank=ranks[0]
                        ),
                        deterministic_cost_s=cost,
                        n_stops=n_stops,
                        stint_lengths=lengths,
                    )
                )

    out.sort(key=lambda c: c.deterministic_cost_s)
    log.info("enumerated %d legal strategies for %s", len(out), params.circuit)
    return out


def optimise(
    params: CircuitParams,
    field: list[Car],
    focus_index: int,
    top_k: int = TOP_K_FOR_SIMULATION,
    n_sims: int = 600,
    seed: int = config.SEED,
    max_stops: int = MAX_STOPS,
) -> pd.DataFrame:
    """Rank strategies for one car: closed form first, then simulated.

    The rest of the field keeps its strategy throughout, so any difference is
    attributable to the car being optimised.
    """
    candidates = enumerate_strategies(params, max_stops=max_stops)
    if not candidates:
        return pd.DataFrame()

    shortlist = candidates[:top_k]
    rows = []
    for cand in shortlist:
        trial = list(field)
        base = field[focus_index]
        trial[focus_index] = Car(
            driver=base.driver,
            pace_offset_s=base.pace_offset_s,
            grid_position=base.grid_position,
            strategy=cand.strategy,
            team=base.team,
        )
        res = simulate(params, trial, n_sims=n_sims, seed=seed)
        pos = res.finish_positions[:, focus_index]
        t = res.finish_times[:, focus_index]
        rows.append(
            {
                "strategy": str(cand.strategy),
                "n_stops": cand.n_stops,
                "stint_lengths": str(cand.stint_lengths),
                "deterministic_cost_s": cand.deterministic_cost_s,
                "sim_mean_race_time_s": float(t.mean()),
                "sim_mean_position": float(pos.mean()),
                "sim_median_position": float(np.median(pos)),
                "position_lo": float(np.percentile(pos, 5)),
                "position_hi": float(np.percentile(pos, 95)),
                "win_rate": float((pos == 1).mean()),
            }
        )

    out = pd.DataFrame(rows)
    out["deterministic_rank"] = out["deterministic_cost_s"].rank().astype(int)
    out["simulated_rank"] = out["sim_mean_position"].rank().astype(int)
    out["rank_moved"] = out["deterministic_rank"] - out["simulated_rank"]
    return out.sort_values("sim_mean_position").reset_index(drop=True)


def ranking_disagreement(opt: pd.DataFrame) -> dict:
    """How much traffic and cautions reorder the closed-form ranking.

    A large disagreement is the point, not a problem: it is the closed form
    being blind to track position and the simulator not being. A
    disagreement of zero would mean the simulator's extra machinery is
    earning nothing, which would itself be worth knowing.
    """
    from scipy import stats

    if len(opt) < 3:
        return {"status": "too_few_candidates", "n": len(opt)}
    rho, p = stats.spearmanr(opt["deterministic_rank"], opt["simulated_rank"])
    best_det = opt.loc[opt["deterministic_rank"].idxmin(), "strategy"]
    best_sim = opt.loc[opt["simulated_rank"].idxmin(), "strategy"]
    return {
        "status": "ok",
        "n_candidates": len(opt),
        "spearman_rho": float(rho),
        "p_value": float(p),
        "best_deterministic": best_det,
        "best_simulated": best_sim,
        "same_winner": bool(best_det == best_sim),
        "max_rank_move": int(opt["rank_moved"].abs().max()),
    }
