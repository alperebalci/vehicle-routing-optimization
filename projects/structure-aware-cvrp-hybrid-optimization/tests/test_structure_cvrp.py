import dataclasses
import itertools
import json
import random
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import pytest

from cvrp_lab.domain import Instance, audit, exact_cost, generated, initial_routes, load
from cvrp_lab.moves import Move, State, propose
from cvrp_lab.mip import build_model, decode, encode_routes, neighborhood, solve_mip
from cvrp_lab.search import metrics, run
from cvrp_lab.adapters import external


def tiny():
    return Instance(np.array([[0, 2, 4, 3], [2, 0, 3, 5], [4, 3, 0, 2], [3, 5, 2, 0]]),
                    np.array([0, 1, 1, 1]), 2, 2)


@pytest.mark.parametrize('seed', range(8))
def test_move_deltas_and_random_rollback(seed):
    p = generated(seed, 30)
    s = State(p, initial_routes(p))
    rng = random.Random(seed)
    for _ in range(600):
        m = propose(s, rng)
        if m is None:
            continue
        ev = s.evaluate(m)
        assert ev == s.evaluate(m, 'full')
        if ev is not None:
            before = s.routes
            token = s.apply(ev)
            s.check()
            if rng.random() < .5:
                s.undo(token)
                assert s.routes == before
                s.check()
            s.commit()


def test_all_small_moves_including_empty_slots():
    p = Instance(tiny().costs, [0, 0, 1, 1], 2, 3)
    s = State(p, [[1, 2], [3]])
    for r, a in enumerate(s.routes):
        for t, b in enumerate(s.routes):
            moves = []
            if r == t:
                moves = [Move('two_opt', r, i, r, j) for i in range(len(a))
                         for j in range(i+2, len(a)+1)]
            else:
                moves += [Move('relocate', r, i, t, j) for i in range(len(a))
                          for j in range(len(b)+1)]
                moves += [Move('swap', r, i, t, j) for i in range(len(a)) for j in range(len(b))]
                moves += [Move('tail', r, i, t, j) for i in range(len(a)+1)
                          for j in range(len(b)+1)]
            for m in moves:
                ev = s.evaluate(m)
                assert ev == s.evaluate(m, 'full')
                if ev is not None:
                    before = s.routes
                    undo = s.apply(ev)
                    s.check()
                    s.undo(undo)
                    assert s.routes == before
                    s.check()


def test_revision_foreign_and_forged_delta():
    p = tiny()
    a, b = State(p, [[1, 2], [3]]), State(p, [[1, 2], [3]])
    m = Move('two_opt', 0, 0, 0, 2)
    ev = a.evaluate(m)
    with pytest.raises(ValueError):
        b.apply(ev)
    with pytest.raises(ValueError):
        a.apply(dataclasses.replace(ev, delta=ev.delta+1))
    token = a.apply(ev)
    with pytest.raises(ValueError):
        a.apply(ev)
    a.undo(token)
    with pytest.raises(ValueError):
        a.apply(ev)
    with pytest.raises(ValueError):
        a.undo(token)


def test_nested_undo_is_lifo():
    s = State(tiny(), [[1, 2], [3]])
    before = s.routes
    m = Move('two_opt', 0, 0, 0, 2)
    first = s.apply(s.evaluate(m))
    second = s.apply(s.evaluate(m))
    with pytest.raises(ValueError):
        s.undo(first)
    s.undo(second)
    s.undo(first)
    s.check()
    assert s.routes == before


@pytest.mark.parametrize('case', ['asym', 'nan', 'negative', 'fractional', 'diagonal', 'shape'])
def test_bad_matrices_rejected(case):
    c = tiny().costs.astype(float).copy()
    if case == 'asym': c[1, 0] += 1
    if case == 'nan': c[0, 1] = np.nan
    if case == 'negative': c[0, 1] = c[1, 0] = -1
    if case == 'fractional': c[0, 1] = c[1, 0] = 1.5
    if case == 'diagonal': c[1, 1] = 1
    if case == 'shape': c = c[:, :-1]
    with pytest.raises(ValueError):
        Instance(c, [0, 1, 1, 1], 2, 2)


@pytest.mark.parametrize('routes', [[[1, 1], [2, 3]], [[1], [2]], [[1, 2, 3]],
                                    [[0, 1], [2, 3]], [[1.0, 2], [3]]])
def test_bad_routes_rejected(routes):
    with pytest.raises(ValueError):
        audit(tiny(), routes)


def test_immutable_input_and_json(tmp_path):
    p = tiny()
    with pytest.raises(ValueError): p.costs[0, 1] = 0
    path = tmp_path/'p.json'
    path.write_text(json.dumps(p.as_dict()))
    assert load(path).fingerprint() == p.fingerprint()


@pytest.mark.parametrize('seed', range(5))
def test_mip_dp_and_encoded_incumbent(seed):
    p = generated(seed, 6)
    init = initial_routes(p)
    m = build_model(p, init)
    z = encode_routes(p, init, m['arcs'])
    az = m['A'] @ z
    assert np.all(az >= m['lo']-1e-6)
    assert np.all(az <= m['hi']+1e-6)
    assert decode(p, m['arcs'], z)[1] == audit(p, init)
    result = solve_mip(p, 3, init)
    assert result['proven_optimal']
    assert abs(result['objective']-exact_cost(p)) < 1e-4
    assert result['lower_bound'] <= result['objective']+1e-4


def brute(p):
    best = float('inf')
    for order in itertools.permutations(range(1, p.n+1)):
        for cut in itertools.product([False, True], repeat=p.n-1):
            routes = [[order[0]]]
            for v, new in zip(order[1:], cut):
                if new: routes.append([])
                routes[-1].append(v)
            try: best = min(best, audit(p, routes))
            except ValueError: pass
    return best


def test_dp_independent_permutation_enumeration():
    assert exact_cost(tiny()) == brute(tiny())


def test_at_most_not_exactly_k():
    p = Instance([[0, 1, 1], [1, 0, 1], [1, 1, 0]], [0, 1, 1], 2, 2)
    r = solve_mip(p, 2)
    assert r['objective'] == 3
    assert len(r['routes']) == 1


def test_zero_demand_connectivity():
    p = Instance([[0, 9, 9], [9, 0, 1], [9, 1, 0]], [0, 0, 0], 1, 1)
    assert solve_mip(p, 2)['objective'] == 19
    assert exact_cost(p) == 19


def test_insufficient_bin_packing_is_infeasible_not_just_total_load():
    p = Instance(tiny().costs, [0, 2, 2, 2], 3, 2)
    assert sum(p.demands) == p.max_vehicles*p.capacity
    with pytest.raises(RuntimeError, match='initialization failed'):
        initial_routes(p)
    r = solve_mip(p, 2)
    assert r['objective'] is None and not r['proven_optimal']
    assert exact_cost(p) == float('inf')


def test_neighborhood_preserves_outside_and_bound_scope():
    p = generated(7, 8)
    initial = [[1, 2], [3, 4], [5, 6, 7, 8]]
    p = Instance(p.costs, np.r_[0, np.ones(8, dtype=int)], 4, 3)
    candidate, detail = neighborhood(p, initial, [0, 1], 2)
    assert initial[2] in candidate
    assert audit(p, candidate) <= audit(p, initial)
    assert detail['bound_scope'] == 'neighborhood'
    assert 'global_lower_bound' not in detail


def test_late_solver_output_is_not_accepted():
    p = tiny()
    init = [[1, 3], [2]]
    model = build_model(p, init)
    from scipy.optimize import OptimizeResult
    z = encode_routes(p, [[1], [2, 3]], model['arcs'])
    def late(*args, **kwargs):
        time.sleep(.06)
        return OptimizeResult(x=z, fun=float(model['c']@z), status=0,
                              message='mock delayed completion', mip_dual_bound=0)
    with patch('cvrp_lab.mip.milp', late):
        result = solve_mip(p, .04, init)
    assert result['objective'] == audit(p, init)
    assert result['late_candidate_objective'] is not None
    assert result['lower_bound'] is None and not result['proven_optimal']


@pytest.mark.parametrize('seed', range(4))
def test_fixed_budget_trajectory_equivalence(seed):
    p = generated(seed, 24)
    a = run(p, 'full', proposals=2000, seed=seed)
    b = run(p, 'incremental', proposals=2000, seed=seed)
    assert a['routes'] == b['routes']
    assert a['accepted'] == b['accepted']
    assert [h['objective'] for h in a['history']] == [h['objective'] for h in b['history']]


@pytest.mark.parametrize('variant', ['full', 'incremental', 'expanded', 'sequential', 'iterative', 'mip'])
def test_wall_budget_and_monotone_audited_history(variant):
    p = generated(4, 6)
    r = run(p, variant, seconds=.08, mip_every=10, mip_slice=.03)
    assert audit(p, r['routes']) == r['objective']
    assert all(h['time'] <= .08 for h in r['history'])
    assert all(a['objective'] >= b['objective'] for a, b in zip(r['history'], r['history'][1:]))
    if variant == 'iterative': assert r['global_lower_bound'] is None


def test_metrics_censoring_and_reference_labels():
    r = {'history': [{'time': 1, 'objective': 15}, {'time': 2, 'objective': 12}],
         'objective': 12, 'budget_seconds': 3, 'elapsed_seconds': 3}
    m = metrics(r, 10, proven=False, target_gap=.1)
    assert m['target_not_reached'] and m['observed_time_to_target'] is None
    assert m['observed_gap_area_after_initialization'] == pytest.approx(.7)
    assert not m['reference_proven_optimal'] and m['no_solution_seconds'] == 1


def test_missing_external_solvers_are_not_runs():
    with patch('cvrp_lab.adapters.importlib.util.find_spec', return_value=None):
        for name in ['gurobi', 'hexaly', 'pyvrp']:
            r = external(tiny(), name)
            assert not r['executed'] and r['objective'] is None
            assert r['status'] == 'not_run_missing_dependency'


def test_cli_micro_json(tmp_path):
    path = tmp_path/'out.json'
    subprocess.run([sys.executable, '-m', 'cvrp_lab.benchmark', '--mode', 'micro',
                    '--sizes', '6', '--seeds', '1', '--proposals', '20', '--out', str(path)], check=True)
    assert json.loads(path.read_text())['micro'][0]['evaluations'] == 20
