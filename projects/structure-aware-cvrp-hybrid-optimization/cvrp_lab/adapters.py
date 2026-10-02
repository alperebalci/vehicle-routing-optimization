"""Optional external comparators. Missing dependencies are explicit, never fabricated runs."""
from __future__ import annotations

import importlib.util
from importlib.metadata import version, PackageNotFoundError
import math
from time import perf_counter

from .domain import audit
from .mip import solve_mip


PACKAGES = {'gurobi': 'gurobipy', 'hexaly': 'hexaly', 'pyvrp': 'pyvrp'}


def availability():
    result = {}
    for name, package in PACKAGES.items():
        present = importlib.util.find_spec(package) is not None
        try:
            ver = version(package) if present else None
        except PackageNotFoundError:
            ver = 'unknown'
        result[name] = {'installed': present, 'version': ver,
                        'license_verified': False, 'executed': False}
    return result


def _hexaly(p, deadline, seed, threads):
    import hexaly.optimizer as hx
    with hx.HexalyOptimizer() as opt:
        model = opt.model
        tours = [model.list(p.n) for _ in range(p.max_vehicles)]
        model.constraint(model.partition(tours))
        demand = model.array([int(v) for v in p.demands[1:]])
        distance = model.array(p.costs[1:, 1:].tolist())
        depot = model.array(p.costs[0, 1:].tolist())
        terms = []
        for tour in tours:
            count = model.count(tour)
            load = model.sum(tour, model.lambda_function(lambda v: demand[v]))
            model.constraint(load <= p.capacity)
            edge_cost = model.lambda_function(
                lambda t: model.at(distance, tour[t-1], tour[t]))
            internal = model.sum(model.range(1, count), edge_cost)
            ends = model.iif(count > 0, depot[tour[0]]+depot[tour[count-1]], 0)
            terms.append(internal+ends)
        # Distance only. No lexicographic vehicle minimization or per-vehicle cost.
        objective = model.sum(terms)
        model.minimize(objective)
        model.close()
        remaining = math.floor(deadline-perf_counter())
        if remaining < 1:
            return None, {'status': 'budget_below_integer_solver_time_limit'}
        opt.param.time_limit = remaining
        opt.param.nb_threads = threads
        opt.param.seed = seed
        opt.param.verbosity = 0
        opt.solve()
        routes = [[int(v)+1 for v in tour.value] for tour in tours]
        # Invalid/no-solution output raises rather than being accepted as feasible.
        cost = audit(p, routes)
        if cost != round(objective.value):
            raise AssertionError('Hexaly objective audit failed')
        return routes, {'status': str(opt.solution.status), 'solver_objective': cost,
                        'thread_limit_is_advisory': True}


def _pyvrp(p, deadline, seed):
    import pyvrp
    from pyvrp.stop import MaxRuntime
    ver = version('pyvrp')
    if not ver.startswith('0.14.'):
        raise RuntimeError('adapter targets PyVRP 0.14.x; validate other APIs before use')
    if p.coordinates is None:
        raise ValueError('PyVRP comparison requires coordinates; no artificial geometry is supplied')
    model = pyvrp.Model()
    locations = [model.add_location(float(x), float(y)) for x, y in p.coordinates]
    depot = model.add_depot(location=locations[0])
    for v in range(1, p.n+1):
        model.add_client(location=locations[v], delivery=int(p.demands[v]))
    model.add_vehicle_type(num_available=p.max_vehicles, capacity=p.capacity,
                           start_depot=depot, end_depot=depot, fixed_cost=0)
    for i, a in enumerate(locations):
        for j, b in enumerate(locations):
            model.add_edge(a, b, distance=int(p.costs[i, j]), duration=0)
    remaining = deadline-perf_counter()
    if remaining <= 0:
        return None, {'status': 'budget_exhausted'}
    result = model.solve(stop=MaxRuntime(remaining-min(.02, .1*remaining)),
                         seed=seed, display=False)
    if not result.best.is_feasible():
        return None, {'status': 'no_feasible_solution'}
    # 0.14 activities use separate zero-based client/depot indices.
    routes = [[int(a.idx)+1 for a in route if not a.is_depot()]
              for route in result.best.routes()]
    cost = audit(p, routes)
    if cost != result.cost():
        raise AssertionError('PyVRP objective audit failed')
    return routes, {'status': 'feasible', 'solver_objective': cost,
                    'coordinates_from_input': True}


def external(p, backend, seconds=10.0, seed=0, focus=0, threads=1):
    if backend not in PACKAGES or seconds <= 0 or not math.isfinite(seconds):
        raise ValueError('invalid external comparator or time budget')
    info = availability()[backend]
    if not info['installed']:
        return {'backend': backend, 'status': 'not_run_missing_dependency',
                'executed': False, 'objective': None, 'version': None, 'seed': seed, 'focus': focus}
    start = perf_counter()
    try:
        if backend == 'gurobi':
            result = solve_mip(p, seconds, backend='gurobi', seed=seed, focus=focus,
                               threads=threads)
            result.update(executed=True, start_protocol='cold', version=info['version'], focus=focus)
            return result
        deadline = start+seconds
        routes, detail = (_hexaly(p, deadline, seed, threads) if backend == 'hexaly'
                          else _pyvrp(p, deadline, seed))
        raw = audit(p, routes) if routes is not None else None
        elapsed = perf_counter()-start
        on_time = elapsed <= seconds
        return dict(detail, backend=backend, executed=True, version=info['version'],
                    start_protocol='cold', routes=routes if on_time else None,
                    objective=raw if on_time else None,
                    late_candidate_objective=raw if not on_time else None,
                    lower_bound=None, bound_scope='not_collected',
                    elapsed_seconds=elapsed, budget_seconds=seconds,
                    overrun_seconds=max(0.0, elapsed-seconds),
                    history_resolution='return_only')
    except Exception as exc:
        # Execution/API/license failures remain visible; never substituted by another solver.
        return {'backend': backend, 'status': 'execution_error', 'executed': True,
                'error_type': type(exc).__name__, 'error': str(exc), 'objective': None,
                'version': info['version'], 'elapsed_seconds': perf_counter()-start}
