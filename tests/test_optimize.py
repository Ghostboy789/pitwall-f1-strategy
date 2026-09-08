"""The optimiser must search, not re-answer the same question k times."""

from __future__ import annotations

import numpy as np

from pitwall.optimize import deterministic_cost, diverse_candidates, enumerate_strategies
from pitwall.sim import CircuitParams


def _params(race_laps=60, **kw) -> CircuitParams:
    base = dict(
        circuit="test",
        race_laps=race_laps,
        base_lap_s=90.0,
        pit_loss_s=22.0,
        fuel_s_per_lap=-0.06,
        caution_hazard_per_lap=0.02,
        traffic_s=1.0,
        deg_by_rank={0: 0.10, 1: 0.06, 2: 0.03},
        pass_p_base=0.08,
    )
    base.update(kw)
    return CircuitParams(**base)


def test_closed_form_matches_the_hand_calculation():
    """A 20/20 split on a compound degrading at 0.1 costs 2 * 0.1 * 20*21/2."""
    cost, lengths = deterministic_cost((20,), (0, 0), 40, {0: 0.10}, pit_loss_s=22.0)
    assert lengths == (20, 20)
    assert np.isclose(cost, 2 * 0.10 * 20 * 21 / 2 + 22.0)


def test_stints_shorter_than_the_minimum_are_rejected():
    cost, _ = deterministic_cost((2,), (0, 1), 40, {0: 0.1, 1: 0.05}, 22.0)
    assert not np.isfinite(cost)


def test_two_compound_regulation_is_enforced():
    """A single-compound strategy would be disqualified, so it must not appear."""
    for c in enumerate_strategies(_params(), max_stops=2):
        ranks = {c.strategy.start_rank, *(r for _, r in c.strategy.stops)}
        assert len(ranks) >= 2


def test_more_stops_are_enumerated_than_one():
    cands = enumerate_strategies(_params(), max_stops=2)
    assert {c.n_stops for c in cands} == {1, 2}


def test_diverse_shortlist_contains_distinct_families():
    """The bug this exists for: the global top k is one plan shifted by a lap."""
    cands = enumerate_strategies(_params(), max_stops=2)
    naive = cands[:6]
    naive_families = {
        (c.n_stops, c.strategy.start_rank, tuple(r for _, r in c.strategy.stops)) for c in naive
    }
    diverse = diverse_candidates(cands, 6)
    diverse_families = {
        (c.n_stops, c.strategy.start_rank, tuple(r for _, r in c.strategy.stops)) for c in diverse
    }
    assert len(diverse_families) == 6, "shortlist repeats a family"
    assert len(diverse_families) > len(naive_families)


def test_diverse_shortlist_keeps_the_cheapest_member_of_each_family():
    cands = enumerate_strategies(_params(), max_stops=2)
    best = diverse_candidates(cands, 4)
    for c in best:
        family = (c.n_stops, c.strategy.start_rank, tuple(r for _, r in c.strategy.stops))
        same = [
            x.deterministic_cost_s
            for x in cands
            if (x.n_stops, x.strategy.start_rank, tuple(r for _, r in x.strategy.stops)) == family
        ]
        assert np.isclose(c.deterministic_cost_s, min(same))


def test_shortlist_is_ordered_by_cost():
    best = diverse_candidates(enumerate_strategies(_params(), max_stops=2), 8)
    costs = [c.deterministic_cost_s for c in best]
    assert costs == sorted(costs)


def test_stint_longer_than_the_observed_ceiling_is_rejected():
    """The feasibility constraint the sanity gate forced.

    A linear degradation model cannot see the tyre cliff, so without a ceiling
    the optimiser proposes stints no team has ever run and "beats" reality by
    the difference.
    """
    caps = {0: 20, 1: 30, 2: 40}
    ok, _ = deterministic_cost((25,), (1, 0), 45, {0: 0.1, 1: 0.06}, 22.0, caps)
    assert np.isfinite(ok), "a 25/20 split is inside the caps and should be allowed"

    bad, lengths = deterministic_cost((44,), (1, 0), 60, {0: 0.1, 1: 0.06}, 22.0, caps)
    assert lengths[0] == 44 > caps[1]
    assert not np.isfinite(bad), "a stint beyond the observed ceiling must be rejected"


def test_no_ceiling_means_no_constraint():
    """Absent limits the cost is unchanged, so the constraint is opt-in."""
    a, _ = deterministic_cost((44,), (1, 0), 60, {0: 0.1, 1: 0.06}, 22.0, None)
    b, _ = deterministic_cost((44,), (1, 0), 60, {0: 0.1, 1: 0.06}, 22.0, {})
    assert np.isfinite(a) and np.isfinite(b)
    assert np.isclose(a, b)


def test_enumeration_respects_the_ceiling():
    p = _params(race_laps=60)
    p.max_stint_by_rank = {0: 15, 1: 20, 2: 25}
    for c in enumerate_strategies(p, max_stops=2):
        ranks = (c.strategy.start_rank, *(r for _, r in c.strategy.stops))
        for n, rank in zip(c.stint_lengths, ranks):
            assert n <= p.max_stint_by_rank[rank], f"{n} laps on rank {rank} exceeds the cap"
