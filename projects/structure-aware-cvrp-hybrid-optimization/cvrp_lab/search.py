"""Matched local-search variants and finite-time MIP hybrid orchestration."""
from __future__ import annotations

import math
import random
from time import perf_counter

from .domain import audit, initial_routes
from .moves import State, propose
from .mip import neighborhood, solve_mip


VARIANTS = {"full", "incremental", "expanded", "sequential", "iterative", "mip"}


def run(p, variant="expanded", *, seed=0, seconds=None, proposals=10000,
        initial=None, backend="highs", mip_slice=.25, mip_every=300):
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("seconds must be positive and finite")
    if proposals <= 0 or mip_slice <= 0 or mip_every <= 0:
        raise ValueError("positive proposal/MIP limits required")
    if variant in {"sequential", "iterative", "mip"} and seconds is None:
        raise ValueError("hybrid/MIP variants require a wall-time budget")
    start = perf_counter()
    deadline = start+seconds if seconds is not None else math.inf
    routes = initial_routes(p) if initial is None else [list(r) for r in initial]
    initial_cost = audit(p, routes)
    state = State(p, routes)
    setup = perf_counter()-start
    best, best_cost = state.routes, state.total
    history = [{"time": setup, "objective": best_cost}] if perf_counter() <= deadline else []
    rng = random.Random(seed)
    attempts = evaluated = accepted = 0
    evaluation_time = application_time = mip_time = 0.0
    mip_calls, global_bound = [], None
    local_deadline = start+seconds*.4 if variant == "sequential" else deadline
    limit = proposals if seconds is None else math.inf
    if variant == "mip":
        local_deadline = start
    mode = "full" if variant == "full" else "incremental"
    expanded = variant not in {"full", "incremental"}
    while attempts < limit and perf_counter() < local_deadline:
        m = propose(state, rng, expanded)
        attempts += 1
        if m is not None:
            tick = perf_counter()
            ev = state.evaluate(m, mode)
            evaluation_time += perf_counter()-tick
            evaluated += ev is not None
            if ev is not None:
                temp = max(1.0, initial_cost*.01 * .9995**attempts)
                take = ev.delta <= 0 or rng.random() < math.exp(-ev.delta/temp)
                if take and perf_counter() < local_deadline:
                    tick = perf_counter()
                    token = state.apply(ev)
                    state.check()
                    snapshot = state.routes if state.total < best_cost else None
                    now = perf_counter()
                    if now > local_deadline:
                        state.undo(token)
                    else:
                        accepted += 1
                        state.commit()
                        if state.total < best_cost:
                            best, best_cost = snapshot, state.total
                            history.append({"time": now-start, "objective": best_cost})
                    application_time += perf_counter()-tick
        if variant == "iterative" and attempts % mip_every == 0 and perf_counter() < deadline:
            tick = perf_counter()
            nonempty = [k for k, route in enumerate(best) if route]
            selected = rng.sample(nonempty, min(2, len(nonempty)))
            candidate, detail = neighborhood(p, best, selected,
                min(mip_slice, max(1e-6, deadline-perf_counter())), backend, seed)
            candidate_cost = audit(p, candidate)
            now = perf_counter()
            detail['accepted_by_parent'] = False
            if now <= deadline and candidate_cost < best_cost:
                candidate_state = State(p, candidate)
                now = perf_counter()
                if now <= deadline:
                    state = candidate_state
                    best, best_cost = state.routes, state.total
                    history.append({"time": now-start, "objective": best_cost})
                    detail['accepted_by_parent'] = True
            mip_calls.append(detail)
            mip_time += perf_counter()-tick
    if variant in {"mip", "sequential"} and perf_counter() < deadline:
        tick = perf_counter()
        detail = solve_mip(p, max(1e-6, deadline-perf_counter()), best, backend, seed)
        now = perf_counter()
        if now <= deadline:
            if detail['objective'] is not None and detail['objective'] < best_cost:
                best, best_cost = tuple(tuple(r) for r in detail['routes']), detail['objective']
                history.append({"time": now-start, "objective": best_cost})
            global_bound = detail['lower_bound']
        detail['accepted_by_parent'] = now <= deadline
        mip_calls.append(detail)
        mip_time += perf_counter()-tick
    audit(p, best)
    elapsed = perf_counter()-start
    return {"variant": variant, "backend": backend, "seed": seed,
            "routes": best, "initial_objective": initial_cost, "objective": best_cost,
            "timed_objective": history[-1]["objective"] if history else None,
            "history": history, "global_lower_bound": global_bound,
            "certified_gap": None if global_bound is None else
                max(0.0, best_cost-global_bound)/max(1, abs(best_cost)),
            "setup_seconds": setup, "evaluation_seconds": evaluation_time,
            "application_audit_seconds": application_time, "mip_seconds": mip_time,
            "elapsed_seconds": elapsed, "budget_seconds": seconds,
            "overrun_seconds": max(0.0, elapsed-seconds) if seconds else 0.0,
            "proposals": attempts, "valid_evaluations": int(evaluated), "accepted": accepted,
            "mip_calls": mip_calls, "timed_initial_available": bool(history),
            "history_resolution": "audited_improvements; MIP result observed only at return"}


def metrics(result, reference, *, proven=False, target_gap=.01):
    """Observed time-to-target and normalized area; reference is never called a bound."""
    if reference <= 0 or not math.isfinite(reference):
        raise ValueError("positive finite reference required")
    horizon = result['budget_seconds'] or result['elapsed_seconds']
    history = [h for h in result['history'] if h['time'] <= horizon]
    hit = next((h['time'] for h in history
                if h['objective'] <= reference*(1+target_gap)), None)
    area, previous, value = 0.0, 0.0, None
    for point in history:
        if value is not None:
            area += (point['time']-previous)*(value-reference)/reference
        value, previous = point['objective'], point['time']
    if value is not None:
        area += (horizon-previous)*(value-reference)/reference
    return {"reference_objective": reference, "reference_proven_optimal": proven,
            "reference_gap": (result['objective']-reference)/reference,
            "observed_time_to_target": hit, "target_not_reached": hit is None,
            "observed_gap_area_after_initialization": area if history else None,
            "no_solution_seconds": history[0]['time'] if history else horizon,
            "integration_horizon_seconds": horizon}
