import random

import pytest

from alns_vrp.alns import ALNSConfig, solve_alns
from alns_vrp.data import CVRPInstance, demo_instance
from alns_vrp.local_search import two_opt
from alns_vrp.solution import Solution


@pytest.mark.parametrize('seed', range(8))
def test_full_incremental_same_route(seed):
    rng = random.Random(seed)
    xy = tuple((rng.random()*100, rng.random()*100) for _ in range(25))
    p = CVRPInstance(xy, (0,)+(1,)*24, 24, 1)
    route = list(p.customers)
    rng.shuffle(route)
    initial = Solution([route])
    a = two_opt(initial, p, evaluation='full')
    b = two_opt(initial, p, evaluation='incremental')
    assert a.routes == b.routes
    assert b.total_distance(p) <= initial.total_distance(p)+1e-9
    assert initial.routes == [route]


@pytest.mark.parametrize('seed', [4, 11, 12])
def test_whole_alns_matched_decisions(seed):
    p = demo_instance(seed, 18)
    a = solve_alns(p, ALNSConfig(iterations=60, local_search_evaluation='full'), seed)
    b = solve_alns(p, ALNSConfig(iterations=60, local_search_evaluation='incremental'), seed)
    assert a.best_solution.routes == b.best_solution.routes
    assert [r.accepted for r in a.records] == [r.accepted for r in b.records]
    assert [r.best_cost for r in a.records] == pytest.approx([r.best_cost for r in b.records])


def test_invalid_evaluator():
    p = demo_instance(1, 4)
    with pytest.raises(ValueError):
        two_opt(Solution([[1, 2, 3, 4]]), p, evaluation='unknown')
    with pytest.raises(ValueError):
        ALNSConfig(local_search_evaluation='unknown').validate(4)
