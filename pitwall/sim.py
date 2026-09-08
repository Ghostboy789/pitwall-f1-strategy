"""Phase 5 - Monte Carlo race simulator.

The mechanic that matters
-------------------------
Most of this is bookkeeping: add up lap times, subtract pit stops. The one
piece that decides whether the whole project says anything is the *blocking*
rule.

A car's raw pace is not what it achieves. If a faster car catches a slower one
and cannot get past, its lap time becomes the slower car's lap time - it
inherits the pace of the thing in front. So each lap, every car that would
have moved ahead of the car in front on raw pace instead draws against the
overtaking model. If the draw fails, its cumulative time is clamped to a car
length behind and it eats a dirty-air penalty.

That clamp is the entire reason track position is worth anything. Remove it
and Monaco and Monza become the same circuit, differing only in lap time.

Safety cars
-----------
Drawn per lap from the circuit's caution hazard. While one is out, lap times
inflate, the field compresses toward even spacing, and a pit stop costs
substantially less because everyone else is slow too. That last effect is the
single largest strategic lever in real racing and the simulator would be
useless without it.

Output is always a distribution. Every entry point returns per-simulation
finishing positions and gaps, never a single deterministic race.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitwall import config

log = logging.getLogger("pitwall.sim")

# Minimum following distance, in seconds. A car that cannot pass sits here.
MIN_GAP_S = 0.7

# Safety car behaviour. The lap-time multiplier and the share of pit loss
# saved are physical constants of the procedure rather than fitted values;
# both are exposed on CircuitParams so a caller can vary them.
SC_LAP_MULTIPLIER = 1.38
SC_PIT_LOSS_FACTOR = 0.45
SC_MIN_LAPS = 3
SC_MAX_LAPS = 6


@dataclass
class CircuitParams:
    """Everything the simulator needs to know about a circuit.

    Every value here is estimated from data by another module, not assumed:
    pace and degradation from ``models.pace`` and ``models.degradation``, pit
    loss and caution hazard from ``models.raceparams``, and the pass
    probability table from ``models.overtaking``.
    """

    circuit: str
    race_laps: int
    base_lap_s: float
    pit_loss_s: float
    fuel_s_per_lap: float
    caution_hazard_per_lap: float
    traffic_s: float
    deg_by_rank: dict[int, float]
    # p(pass per lap) indexed by pace advantage in s/lap; see pass_probability.
    pass_p_base: float
    pass_p_per_s: float = 0.35
    lap_time_sd_s: float = 0.25
    sc_lap_multiplier: float = SC_LAP_MULTIPLIER
    sc_pit_loss_factor: float = SC_PIT_LOSS_FACTOR

    def pass_probability(self, pace_advantage_s: np.ndarray) -> np.ndarray:
        """Per-lap probability of completing a pass, given a pace advantage.

        A logistic in the follower's pace advantage, anchored on the circuit's
        fitted base rate. The shape is deliberately simple: the simulator
        evaluates this millions of times, and the circuit-to-circuit
        difference - which is what the project is measuring - lives in
        ``pass_p_base``, which comes straight from the hierarchical model.
        """
        base = np.clip(self.pass_p_base, 1e-4, 0.95)
        logit = np.log(base / (1 - base)) + self.pass_p_per_s * np.maximum(
            pace_advantage_s, 0.0
        )
        return 1.0 / (1.0 + np.exp(-logit))


@dataclass
class Strategy:
    """A planned set of stops: (lap to pit on, compound rank fitted)."""

    stops: tuple[tuple[int, int], ...]
    start_rank: int = 0

    def compound_at(self, lap: int) -> int:
        rank = self.start_rank
        for stop_lap, stop_rank in self.stops:
            if lap > stop_lap:
                rank = stop_rank
        return rank

    def __str__(self) -> str:
        return "start-r%d " % self.start_rank + " ".join(
            f"L{l}->r{r}" for l, r in self.stops
        )


@dataclass
class Car:
    """One entry: who they are, how quick, and what they intend to do."""

    driver: str
    pace_offset_s: float
    grid_position: int
    strategy: Strategy
    team: str = ""


@dataclass
class RaceResult:
    """Per-simulation outcome, plus the inputs that produced it."""

    finish_positions: np.ndarray   # (n_sims, n_cars)
    finish_times: np.ndarray       # (n_sims, n_cars) cumulative seconds
    gaps_to_winner: np.ndarray     # (n_sims, n_cars)
    n_cautions: np.ndarray         # (n_sims,)
    cars: list[Car] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Mean finishing position and gap per car, with intervals."""
        rows = []
        for i, car in enumerate(self.cars):
            pos = self.finish_positions[:, i]
            gap = self.gaps_to_winner[:, i]
            rows.append(
                {
                    "driver": car.driver,
                    "grid": car.grid_position,
                    "strategy": str(car.strategy),
                    "mean_position": float(pos.mean()),
                    "median_position": float(np.median(pos)),
                    "p05_position": float(np.percentile(pos, 5)),
                    "p95_position": float(np.percentile(pos, 95)),
                    "win_rate": float((pos == 1).mean()),
                    "podium_rate": float((pos <= 3).mean()),
                    "mean_gap_s": float(gap.mean()),
                    "gap_lo": float(np.percentile(gap, 5)),
                    "gap_hi": float(np.percentile(gap, 95)),
                }
            )
        return pd.DataFrame(rows).sort_values("mean_position").reset_index(drop=True)


def simulate(
    params: CircuitParams,
    cars: list[Car],
    n_sims: int = config.N_SIM_DEFAULT,
    seed: int = config.SEED,
) -> RaceResult:
    """Run ``n_sims`` races and return the distribution of outcomes.

    Vectorised across simulations: state is an ``(n_sims, n_cars)`` array
    advanced one lap at a time, so 2000 races of 20 cars costs a few hundred
    numpy operations rather than 2.4 million Python iterations.
    """
    rng = np.random.default_rng(seed)
    n_cars = len(cars)
    laps = params.race_laps

    cum = np.zeros((n_sims, n_cars))
    tyre_age = np.zeros((n_sims, n_cars), dtype=int)
    rank_now = np.array([c.strategy.start_rank for c in cars])[None, :].repeat(n_sims, 0)
    pace = np.array([c.pace_offset_s for c in cars])[None, :]

    # Grid position becomes a starting time offset: roughly the gap a grid
    # slot is worth off the line, so the front row does not begin level.
    grid = np.array([c.grid_position for c in cars])[None, :]
    cum += (grid - 1) * 0.35

    sc_laps_left = np.zeros(n_sims, dtype=int)
    n_cautions = np.zeros(n_sims, dtype=int)

    # On-track order at the start of the race is the grid.
    order_prev = np.argsort(cum, axis=1)

    # Pre-resolve each car's planned stop laps and the compound taken.
    stop_lap = [dict(c.strategy.stops) for c in cars]

    for lap in range(1, laps + 1):
        # --- caution state -------------------------------------------------
        starting = (rng.random(n_sims) < params.caution_hazard_per_lap) & (sc_laps_left == 0)
        n_cautions += starting
        sc_laps_left = np.where(
            starting,
            rng.integers(SC_MIN_LAPS, SC_MAX_LAPS + 1, size=n_sims),
            sc_laps_left,
        )
        under_sc = sc_laps_left > 0

        # --- raw lap time --------------------------------------------------
        tyre_age += 1
        deg_rate = np.zeros((n_sims, n_cars))
        for r, rate in params.deg_by_rank.items():
            deg_rate[rank_now == r] = rate

        fuel_gain = params.fuel_s_per_lap * (lap - 1)
        lap_time = (
            params.base_lap_s
            + pace
            + fuel_gain
            + deg_rate * tyre_age
            + rng.normal(0.0, params.lap_time_sd_s, size=(n_sims, n_cars))
        )
        lap_time = np.where(under_sc[:, None], lap_time * params.sc_lap_multiplier, lap_time)

        # --- pit stops -----------------------------------------------------
        for i, car in enumerate(cars):
            if lap in stop_lap[i]:
                cost = np.where(
                    under_sc, params.pit_loss_s * params.sc_pit_loss_factor, params.pit_loss_s
                )
                lap_time[:, i] += cost
                tyre_age[:, i] = 0
                rank_now[:, i] = stop_lap[i][lap]

        cum = cum + lap_time

        # --- blocking: the mechanic that makes track position worth anything
        #
        # Resolved against the order at the START of this lap, not the order
        # implied by the times just computed. A car can only be held up by
        # whoever was actually in front of it when the lap began; using the
        # post-lap order asks whether the car that already got past was
        # blocked by the car it passed, which is backwards and lets every
        # faster car through unimpeded.
        #
        # Front to back, so a blocked car in turn blocks the one behind it and
        # queues form the way they do in a real race.
        rows = np.arange(n_sims)
        for slot in range(1, n_cars):
            ahead = order_prev[:, slot - 1]
            behind = order_prev[:, slot]
            t_ahead = cum[rows, ahead]
            t_behind = cum[rows, behind]

            caught = t_behind < t_ahead + MIN_GAP_S
            if not caught.any():
                continue

            # How much quicker the follower was over this lap.
            advantage = lap_time[rows, ahead] - lap_time[rows, behind]
            p = params.pass_probability(advantage)
            p = np.where(under_sc, 0.0, p)   # nobody overtakes under caution
            got_through = rng.random(n_sims) < p

            blocked = caught & ~got_through
            if blocked.any():
                # Held to the car in front, plus a dirty-air penalty.
                cum[rows[blocked], behind[blocked]] = (
                    t_ahead[blocked] + MIN_GAP_S + params.traffic_s * 0.1
                )

        order_prev = np.argsort(cum, axis=1)
        sc_laps_left = np.maximum(sc_laps_left - 1, 0)

    finish_order = np.argsort(cum, axis=1)
    positions = np.empty_like(finish_order)
    np.put_along_axis(
        positions, finish_order, np.arange(1, n_cars + 1)[None, :].repeat(n_sims, 0), axis=1
    )
    winner_time = cum.min(axis=1, keepdims=True)

    return RaceResult(
        finish_positions=positions,
        finish_times=cum,
        gaps_to_winner=cum - winner_time,
        n_cautions=n_cautions,
        cars=list(cars),
    )


def compare_strategies(
    params: CircuitParams,
    cars: list[Car],
    focus_index: int,
    alternatives: list[Strategy],
    n_sims: int = config.N_SIM_DEFAULT,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Re-run the same race with one car's strategy swapped, everything else held.

    The comparison every counterfactual in this project rests on. The rest of
    the field keeps its actual strategy, so the difference is attributable to
    the one change rather than to a different race.
    """
    rows = []
    for alt in alternatives:
        trial = list(cars)
        trial[focus_index] = Car(
            driver=cars[focus_index].driver,
            pace_offset_s=cars[focus_index].pace_offset_s,
            grid_position=cars[focus_index].grid_position,
            strategy=alt,
            team=cars[focus_index].team,
        )
        res = simulate(params, trial, n_sims=n_sims, seed=seed)
        pos = res.finish_positions[:, focus_index]
        gap = res.gaps_to_winner[:, focus_index]
        rows.append(
            {
                "strategy": str(alt),
                "mean_position": float(pos.mean()),
                "median_position": float(np.median(pos)),
                "position_lo": float(np.percentile(pos, 5)),
                "position_hi": float(np.percentile(pos, 95)),
                "mean_gap_s": float(gap.mean()),
                "gap_lo": float(np.percentile(gap, 5)),
                "gap_hi": float(np.percentile(gap, 95)),
                "win_rate": float((pos == 1).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_position").reset_index(drop=True)
