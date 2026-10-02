"""Arc MILP with delivery flow and visit flow; neighborhood bounds stay local."""
from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace
import math
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .domain import Instance, audit


def encode_routes(p, routes, arcs):
    """Encode a feasible route solution, including both continuous flows."""
    audit(p, routes)
    index, n = {arc: k for k, arc in enumerate(arcs)}, len(arcs)
    values = np.zeros(3*n)
    for route in routes:
        load, visits, prev = sum(int(p.demands[v]) for v in route), len(route), 0
        for v in (*route, 0):
            if v == prev:
                continue
            k = index[prev, v]
            values[k], values[n+k], values[2*n+k] = 1, load, visits
            if v:
                load -= int(p.demands[v])
                visits -= 1
            prev = v
    return values


def build_model(p, incumbent=None):
    arcs = [(i, j) for i in range(p.n+1) for j in range(p.n+1) if i != j]
    count = len(arcs)
    rows, cols, data, low, high = [], [], [], [], []

    def add(terms, lo=-np.inf, hi=np.inf):
        row = len(low)
        for col, value in terms:
            rows.append(row)
            cols.append(col)
            data.append(value)
        low.append(lo)
        high.append(hi)

    incoming = {v: [k for k, (_, j) in enumerate(arcs) if j == v] for v in range(p.n+1)}
    outgoing = {v: [k for k, (i, _) in enumerate(arcs) if i == v] for v in range(p.n+1)}
    for v in range(1, p.n+1):
        add([(k, 1) for k in incoming[v]], 1, 1)
        add([(k, 1) for k in outgoing[v]], 1, 1)
        for offset, demand in ((count, int(p.demands[v])), (2*count, 1)):
            terms = [(offset+k, 1) for k in incoming[v]]
            terms += [(offset+k, -1) for k in outgoing[v]]
            add(terms, demand, demand)
    add([(k, 1) for k in outgoing[0]] + [(k, -1) for k in incoming[0]], 0, 0)
    add([(k, 1) for k in outgoing[0]], 1, p.max_vehicles)
    upper = np.r_[np.ones(count), np.full(count, p.capacity), np.full(count, p.n)]
    for k, (i, j) in enumerate(arcs):
        # Delivery capacity and a separate unit flow exclude zero-demand subtours.
        add([(count+k, 1), (k, -(p.capacity-int(p.demands[i])))], hi=0)
        add([(count+k, 1), (k, -int(p.demands[j]))], lo=0)
        add([(2*count+k, 1), (k, -(p.n-int(i != 0)))], hi=0)
        if j:
            add([(2*count+k, 1), (k, -1)], lo=0)
        else:
            upper[count+k] = upper[2*count+k] = 0
    objective = np.r_[[p.costs[i, j] for i, j in arcs], np.zeros(2*count)]
    if incumbent is not None:
        ub = audit(p, incumbent)
        add([(k, objective[k]) for k in range(count)], hi=ub+1e-6)
    matrix = coo_matrix((data, (rows, cols)), shape=(len(low), 3*count)).tocsc()
    return {"arcs": arcs, "c": objective, "A": matrix,
            "lo": np.array(low), "hi": np.array(high), "upper": upper,
            "integer": np.r_[np.ones(count), np.zeros(2*count)]}


def decode(p, arcs, values):
    if values is None or not np.isfinite(values).all():
        raise ValueError("invalid solver vector")
    x = np.asarray(values[:len(arcs)])
    if np.max(np.abs(x-np.rint(x))) > 1e-5:
        raise ValueError("fractional arc solution")
    edges = [arc for arc, value in zip(arcs, x) if value > .5]
    successors = {}
    for i, j in edges:
        if i and i in successors:
            raise ValueError("multiple outgoing customer arcs")
        if i:
            successors[i] = j
    routes, seen = [], set()
    for _, start in sorted(arc for arc in edges if arc[0] == 0):
        route, v = [], start
        while v:
            if v in seen or v not in successors:
                raise ValueError("disconnected or repeated customer")
            seen.add(v)
            route.append(v)
            v = successors[v]
        routes.append(route)
    objective = audit(p, routes)
    if len(edges) != p.n+len(routes):
        raise ValueError("unexpected arcs")
    return routes, objective


def _gurobi(model, p, incumbent, deadline, seed, focus, threads):
    import gurobipy as gp
    with gp.Model("cvrp_flow") as g:
        g.Params.OutputFlag = 0
        g.Params.Threads = threads
        g.Params.Seed = seed
        g.Params.MIPFocus = focus
        z = g.addMVar(len(model['c']), lb=0, ub=model['upper'], obj=model['c'],
                     vtype=np.where(model['integer'] > 0, 'B', 'C'))
        matrix = model['A'].tocsr()
        eq = model['lo'] == model['hi']
        if np.any(eq):
            g.addMConstr(matrix[eq], z, '=', model['hi'][eq])
        hi = np.isfinite(model['hi']) & ~eq
        lo = np.isfinite(model['lo']) & ~eq
        if np.any(hi):
            g.addMConstr(matrix[hi], z, '<', model['hi'][hi])
        if np.any(lo):
            g.addMConstr(matrix[lo], z, '>', model['lo'][lo])
        if incumbent is not None:
            z.Start = encode_routes(p, incumbent, model['arcs'])
        remaining = deadline-perf_counter()
        if remaining <= 0:
            return None
        g.Params.TimeLimit = remaining - min(.02, .1*remaining)
        g.optimize()
        return SimpleNamespace(status=int(g.Status), message=f"Gurobi status {g.Status}",
                               x=np.array(z.X) if g.SolCount else None,
                               fun=float(g.ObjVal) if g.SolCount else None,
                               mip_dual_bound=float(g.ObjBound),
                               proven_optimal=g.Status == gp.GRB.OPTIMAL)


def solve_mip(p: Instance, seconds=1.0, incumbent=None, backend="highs", seed=0,
              focus=0, threads=1, scope="global"):
    if not math.isfinite(seconds) or seconds <= 0 or threads <= 0:
        raise ValueError("positive finite budget and threads required")
    if backend not in {"highs", "gurobi"} or scope not in {"global", "neighborhood"}:
        raise ValueError("invalid backend or bound scope")
    start = perf_counter()
    deadline = start+seconds
    best = [list(r) for r in incumbent if r] if incumbent is not None else None
    objective = audit(p, best) if best is not None else None
    model = build_model(p, best)
    build_seconds = perf_counter()-start
    result = None
    solve_start = perf_counter()
    if deadline > solve_start:
        if backend == "highs":
            # SciPy's milp API has no MIP-start argument. The incumbent supplies
            # an objective cutoff and fallback, not a solver-internal warm start.
            remaining = deadline-perf_counter()
            if remaining > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Unrecognized options detected.*threads")
                    result = milp(model['c'], integrality=model['integer'],
                                  bounds=Bounds(0, model['upper']),
                                  constraints=LinearConstraint(model['A'], model['lo'], model['hi']),
                                  options={"time_limit": remaining-min(.02, .1*remaining),
                                           "mip_rel_gap": 0, "threads": threads})
                result.proven_optimal = result.status == 0
        else:
            result = _gurobi(model, p, best, deadline, seed, focus, threads)
    solve_seconds = perf_counter()-solve_start
    raw_cost, bound, raw_routes = None, None, None
    if result is not None:
        if result.x is not None:
            raw_routes, raw_cost = decode(p, model['arcs'], result.x)
            if not np.isclose(raw_cost, result.fun, atol=1e-4, rtol=1e-8):
                raise AssertionError("solver objective disagrees with independent audit")
        raw_bound = getattr(result, 'mip_dual_bound', None)
        if raw_bound is not None and np.isfinite(raw_bound):
            bound = max(0.0, float(raw_bound))
    elapsed = perf_counter()-start
    on_time = elapsed <= seconds
    if on_time and raw_cost is not None and (objective is None or raw_cost < objective):
        objective, best = raw_cost, raw_routes
    timed_bound = bound if on_time else None
    if timed_bound is not None and objective is not None and timed_bound > objective+1e-4:
        raise AssertionError("invalid bound above feasible incumbent")
    return {"backend": backend, "seed": seed, "threads_requested": threads,
            "model_variables": len(model["c"]), "binary_variables": len(model["arcs"]),
            "model_rows": model["A"].shape[0], "model_nonzeros": model["A"].nnz,
            "status": str(result.message) if result is not None else "budget_exhausted",
            "routes": best, "objective": objective, "lower_bound": timed_bound,
            "bound_scope": scope, "proven_optimal": bool(on_time and result is not None
                and result.proven_optimal and raw_cost is not None),
            "build_seconds": build_seconds, "solve_seconds": solve_seconds,
            "audit_seconds": max(0.0, elapsed-build_seconds-solve_seconds),
            "elapsed_seconds": elapsed, "budget_seconds": seconds,
            "overrun_seconds": max(0.0, elapsed-seconds),
            "late_candidate_objective": raw_cost if not on_time else None,
            "late_lower_bound": bound if not on_time else None,
            "start_semantics": "MIP_start" if backend == "gurobi" else "cutoff_and_retained_incumbent",
            "history_resolution": "initial_and_return_only"}


def neighborhood(p, routes, selected, seconds, backend="highs", seed=0):
    audit(p, routes)
    if not selected or len(set(selected)) != len(selected) or any(
            k < 0 or k >= len(routes) for k in selected):
        raise ValueError("invalid selected route indices")
    outside = [list(r) for k, r in enumerate(routes) if k not in selected and r]
    ids = [0] + sorted(v for k in selected for v in routes[k])
    if len(ids) == 1:
        raise ValueError("empty neighborhood")
    mapping = {v: i for i, v in enumerate(ids)}
    sub = Instance(p.costs[np.ix_(ids, ids)], p.demands[ids], p.capacity,
                   p.max_vehicles-len(outside), p.name+"-neighborhood")
    init = [[mapping[v] for v in routes[k]] for k in selected if routes[k]]
    result = solve_mip(sub, seconds, init, backend, seed, scope="neighborhood")
    result['selected_customers'] = ids[1:]
    if result['routes'] is not None:
        combined = outside+[[ids[v] for v in r] for r in result['routes']]
        audit(p, combined)
    else:
        combined = [list(r) for r in routes]
    # result['lower_bound'] refers ONLY to the selected-customer subproblem.
    return combined, result
