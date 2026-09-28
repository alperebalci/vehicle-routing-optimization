"""Exact small-instance branch-price-and-cut for a route-based CVRPTW master."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from math import ceil

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


@dataclass(frozen=True)
class VRPTWInstance:
    """Depot-first CVRPTW data."""

    coordinates: NDArray[np.float64]
    demand: NDArray[np.float64]
    ready: NDArray[np.float64]
    due: NDArray[np.float64]
    service: NDArray[np.float64]
    capacity: float
    max_vehicles: int

    @classmethod
    def from_arrays(
        cls,
        coordinates: ArrayLike,
        demand: ArrayLike,
        ready: ArrayLike,
        due: ArrayLike,
        service: ArrayLike,
        capacity: float,
        max_vehicles: int,
    ) -> VRPTWInstance:
        coords = np.asarray(coordinates, dtype=float)
        demand_arr = np.asarray(demand, dtype=float)
        ready_arr = np.asarray(ready, dtype=float)
        due_arr = np.asarray(due, dtype=float)
        service_arr = np.asarray(service, dtype=float)
        n_nodes = coords.shape[0] if coords.ndim == 2 else -1

        if coords.ndim != 2 or coords.shape[1] != 2 or n_nodes < 2:
            raise ValueError("coordinates must be a (depot + customers, 2) matrix")
        for name, arr in (
            ("demand", demand_arr),
            ("ready", ready_arr),
            ("due", due_arr),
            ("service", service_arr),
        ):
            if arr.shape != (n_nodes,) or np.any(~np.isfinite(arr)):
                raise ValueError(f"{name} must be finite with one value per node")
        if np.any(demand_arr < 0.0) or demand_arr[0] != 0.0:
            raise ValueError("demands must be non-negative and depot demand must be zero")
        if np.any(service_arr < 0.0) or np.any(ready_arr > due_arr):
            raise ValueError("invalid service times or time windows")
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("capacity must be positive and finite")
        if max_vehicles < 1:
            raise ValueError("max_vehicles must be positive")

        return cls(
            coords,
            demand_arr,
            ready_arr,
            due_arr,
            service_arr,
            float(capacity),
            int(max_vehicles),
        )

    @property
    def n_customers(self) -> int:
        return int(self.coordinates.shape[0] - 1)

    @property
    def distances(self) -> NDArray[np.float64]:
        delta = self.coordinates[:, None, :] - self.coordinates[None, :, :]
        return np.sqrt(np.sum(delta**2, axis=2))


@dataclass(frozen=True)
class Route:
    customers: tuple[int, ...]
    cost: float

    @property
    def customer_set(self) -> frozenset[int]:
        return frozenset(self.customers)


@dataclass(frozen=True)
class NodeRestrictions:
    together: tuple[tuple[int, int], ...] = ()
    separate: tuple[tuple[int, int], ...] = ()
    forbidden_routes: frozenset[int] = frozenset()
    forced_routes: frozenset[int] = frozenset()


@dataclass
class MasterSolution:
    objective: float
    route_values: dict[int, float]
    artificial: NDArray[np.float64]
    cut_artificial: float
    active_routes: tuple[int, ...]
    dual_customer: NDArray[np.float64]
    dual_vehicle: float
    dual_cut: float


@dataclass(frozen=True)
class BPCResult:
    objective: float
    selected_route_ids: tuple[int, ...]
    selected_routes: tuple[Route, ...]
    nodes_processed: int
    columns_generated: int
    optimal: bool
    oracle_objective: float


def _route_schedule(instance: VRPTWInstance, order: tuple[int, ...]) -> float | None:
    dist = instance.distances
    load = float(np.sum(instance.demand[list(order)])) if order else 0.0
    if load > instance.capacity + 1e-10:
        return None

    time = max(0.0, float(instance.ready[0]))
    previous = 0
    travel_cost = 0.0
    for customer in order:
        travel = float(dist[previous, customer])
        travel_cost += travel
        arrival = time + travel
        start = max(arrival, float(instance.ready[customer]))
        if start > float(instance.due[customer]) + 1e-10:
            return None
        time = start + float(instance.service[customer])
        previous = customer

    travel_back = float(dist[previous, 0])
    travel_cost += travel_back
    if time + travel_back > float(instance.due[0]) + 1e-10:
        return None
    return travel_cost


def enumerate_routes(instance: VRPTWInstance) -> list[Route]:
    """Enumerate the cheapest feasible elementary route for each customer subset."""

    customers = range(1, instance.n_customers + 1)
    best: dict[frozenset[int], Route] = {}

    for size in range(1, instance.n_customers + 1):
        for subset in combinations(customers, size):
            if float(np.sum(instance.demand[list(subset)])) > instance.capacity + 1e-10:
                continue
            best_route: Route | None = None
            for order in permutations(subset):
                cost = _route_schedule(instance, order)
                if cost is None:
                    continue
                candidate = Route(tuple(order), float(cost))
                if best_route is None or candidate.cost < best_route.cost - 1e-12:
                    best_route = candidate
            if best_route is not None:
                best[frozenset(subset)] = best_route

    routes = sorted(
        best.values(),
        key=lambda r: (len(r.customers), tuple(sorted(r.customer_set)), r.cost),
    )
    covered = set().union(*(route.customer_set for route in routes)) if routes else set()
    required = set(customers)
    if covered != required:
        missing = sorted(required - covered)
        raise ValueError(f"customers without any feasible route: {missing}")
    return routes


def _compatible(
    route_id: int,
    route: Route,
    routes: list[Route],
    restrictions: NodeRestrictions,
) -> bool:
    if route_id in restrictions.forbidden_routes:
        return False

    subset = route.customer_set
    for i, j in restrictions.separate:
        if i in subset and j in subset:
            return False
    for i, j in restrictions.together:
        if (i in subset) != (j in subset):
            return False

    forced_customers: set[int] = set()
    for forced_id in restrictions.forced_routes:
        forced_customers.update(routes[forced_id].customer_set)
    return not (
        route_id not in restrictions.forced_routes
        and bool(subset.intersection(forced_customers))
    )


def _initial_active_routes(
    instance: VRPTWInstance,
    routes: list[Route],
    restrictions: NodeRestrictions,
) -> list[int]:
    active = set(restrictions.forced_routes)
    for customer in range(1, instance.n_customers + 1):
        if any(customer in routes[r].customer_set for r in restrictions.forced_routes):
            continue
        candidates = [
            rid
            for rid, route in enumerate(routes)
            if customer in route.customer_set and _compatible(rid, route, routes, restrictions)
        ]
        if candidates:
            active.add(min(candidates, key=lambda rid: (len(routes[rid].customers), routes[rid].cost)))
    return sorted(active)


def _solve_master(
    instance: VRPTWInstance,
    routes: list[Route],
    active_routes: list[int],
    restrictions: NodeRestrictions,
    use_fleet_cut: bool,
) -> MasterSolution:
    n = instance.n_customers
    m = len(active_routes)
    artificial_penalty = 1e6

    extra_cut_var = 1 if use_fleet_cut else 0
    total_vars = m + n + extra_cut_var
    c = np.concatenate(
        [
            np.asarray([routes[r].cost for r in active_routes], dtype=float),
            np.full(n, artificial_penalty, dtype=float),
            np.full(extra_cut_var, artificial_penalty, dtype=float),
        ]
    )

    a_eq = np.zeros((n, total_vars), dtype=float)
    for col, rid in enumerate(active_routes):
        for customer in routes[rid].customer_set:
            a_eq[customer - 1, col] = 1.0
    a_eq[:, m : m + n] = np.eye(n)

    a_ub_rows: list[np.ndarray] = []
    b_ub: list[float] = []

    vehicle = np.zeros(total_vars, dtype=float)
    vehicle[:m] = 1.0
    a_ub_rows.append(vehicle)
    b_ub.append(float(instance.max_vehicles))

    if use_fleet_cut:
        fleet_lb = ceil(float(np.sum(instance.demand[1:])) / instance.capacity - 1e-12)
        cut = np.zeros(total_vars, dtype=float)
        cut[:m] = -1.0
        cut[-1] = -1.0
        a_ub_rows.append(cut)
        b_ub.append(float(-fleet_lb))

    bounds: list[tuple[float, float | None]] = [(0.0, None)] * total_vars
    forced = set(restrictions.forced_routes)
    for col, rid in enumerate(active_routes):
        if rid in forced:
            bounds[col] = (1.0, 1.0)

    result = linprog(
        c=c,
        A_ub=np.stack(a_ub_rows),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=np.ones(n),
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"restricted master failed: {result.message}")

    route_values = {
        rid: float(result.x[col]) for col, rid in enumerate(active_routes) if result.x[col] > 1e-12
    }
    ub_duals = np.asarray(result.ineqlin.marginals, dtype=float)
    return MasterSolution(
        objective=float(result.fun),
        route_values=route_values,
        artificial=np.asarray(result.x[m : m + n], dtype=float),
        cut_artificial=float(result.x[-1]) if use_fleet_cut else 0.0,
        active_routes=tuple(active_routes),
        dual_customer=np.asarray(result.eqlin.marginals, dtype=float),
        dual_vehicle=float(ub_duals[0]),
        dual_cut=float(ub_duals[1]) if use_fleet_cut else 0.0,
    )


def _reduced_cost(
    route: Route,
    master: MasterSolution,
    use_fleet_cut: bool,
) -> float:
    customer_dual = sum(master.dual_customer[i - 1] for i in route.customer_set)
    reduced = route.cost - customer_dual - master.dual_vehicle
    if use_fleet_cut:
        reduced += master.dual_cut
    return float(reduced)


def _column_generation(
    instance: VRPTWInstance,
    routes: list[Route],
    restrictions: NodeRestrictions,
    *,
    use_fleet_cut: bool,
    tolerance: float = 1e-9,
) -> tuple[MasterSolution | None, int]:
    active = _initial_active_routes(instance, routes, restrictions)
    generated = 0

    while True:
        master = _solve_master(instance, routes, active, restrictions, use_fleet_cut)
        active_set = set(active)
        candidates = [
            (rid, _reduced_cost(route, master, use_fleet_cut))
            for rid, route in enumerate(routes)
            if rid not in active_set and _compatible(rid, route, routes, restrictions)
        ]
        negative = [(rid, rc) for rid, rc in candidates if rc < -tolerance]
        if not negative:
            if np.any(master.artificial > 1e-7) or master.cut_artificial > 1e-7:
                return None, generated
            return master, generated

        rid, _ = min(negative, key=lambda item: item[1])
        active.append(rid)
        active.sort()
        generated += 1


def solve_root_column_generation(
    instance: VRPTWInstance,
    routes: list[Route] | None = None,
    *,
    use_fleet_cut: bool = True,
) -> MasterSolution:
    route_list = enumerate_routes(instance) if routes is None else routes
    master, _ = _column_generation(
        instance,
        route_list,
        NodeRestrictions(),
        use_fleet_cut=use_fleet_cut,
    )
    if master is None:
        raise RuntimeError("root master is infeasible")
    return master


def _fractional_pair(
    instance: VRPTWInstance,
    routes: list[Route],
    master: MasterSolution,
    tolerance: float,
) -> tuple[int, int] | None:
    best: tuple[float, int, int] | None = None
    for i in range(1, instance.n_customers + 1):
        for j in range(i + 1, instance.n_customers + 1):
            value = sum(
                lam
                for rid, lam in master.route_values.items()
                if i in routes[rid].customer_set and j in routes[rid].customer_set
            )
            if tolerance < value < 1.0 - tolerance:
                candidate = (abs(value - 0.5), i, j)
                if best is None or candidate < best:
                    best = candidate
    return None if best is None else (best[1], best[2])


def _fractional_route(master: MasterSolution, tolerance: float) -> int | None:
    fractional = [
        (abs(value - 0.5), rid)
        for rid, value in master.route_values.items()
        if tolerance < value < 1.0 - tolerance
    ]
    return min(fractional)[1] if fractional else None


def full_integer_master(instance: VRPTWInstance, routes: list[Route] | None = None) -> tuple[float, tuple[int, ...]]:
    """Solve the complete route master as an independent finite-universe oracle."""

    route_list = enumerate_routes(instance) if routes is None else routes
    n = instance.n_customers
    m = len(route_list)
    a_eq = np.zeros((n, m), dtype=float)
    for rid, route in enumerate(route_list):
        for customer in route.customer_set:
            a_eq[customer - 1, rid] = 1.0

    fleet_lb = ceil(float(np.sum(instance.demand[1:])) / instance.capacity - 1e-12)
    a_ub = np.stack([np.ones(m), -np.ones(m)])
    b_ub = np.array([instance.max_vehicles, -fleet_lb], dtype=float)

    constraints = [
        LinearConstraint(a_eq, np.ones(n), np.ones(n)),
        LinearConstraint(a_ub, -np.inf * np.ones(2), b_ub),
    ]
    result = milp(
        c=np.asarray([route.cost for route in route_list]),
        integrality=np.ones(m, dtype=int),
        bounds=Bounds(np.zeros(m), np.ones(m)),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"full integer master failed: {result.message}")
    selected = tuple(int(i) for i, value in enumerate(result.x) if value > 0.5)
    return float(result.fun), selected


def branch_price_cut(
    instance: VRPTWInstance,
    *,
    use_fleet_cut: bool = True,
    node_limit: int = 1000,
    tolerance: float = 1e-8,
) -> BPCResult:
    """Solve the finite route universe exactly with column generation and branching."""

    routes = enumerate_routes(instance)
    oracle_objective, _ = full_integer_master(instance, routes)

    incumbent = float("inf")
    incumbent_ids: tuple[int, ...] = ()
    nodes = 0
    columns_generated = 0
    stack = [NodeRestrictions()]

    while stack and nodes < node_limit:
        restrictions = stack.pop()
        nodes += 1
        master, generated = _column_generation(
            instance,
            routes,
            restrictions,
            use_fleet_cut=use_fleet_cut,
            tolerance=tolerance,
        )
        columns_generated += generated
        if master is None or master.objective >= incumbent - tolerance:
            continue

        fractional_pair = _fractional_pair(instance, routes, master, tolerance)
        fractional_route = _fractional_route(master, tolerance)
        if fractional_pair is None and fractional_route is None:
            selected = tuple(sorted(rid for rid, value in master.route_values.items() if value > 0.5))
            objective = float(sum(routes[rid].cost for rid in selected))
            if objective < incumbent - tolerance:
                incumbent = objective
                incumbent_ids = selected
            continue

        if fractional_pair is not None:
            pair = fractional_pair
            stack.append(
                NodeRestrictions(
                    together=restrictions.together,
                    separate=restrictions.separate + (pair,),
                    forbidden_routes=restrictions.forbidden_routes,
                    forced_routes=restrictions.forced_routes,
                )
            )
            stack.append(
                NodeRestrictions(
                    together=restrictions.together + (pair,),
                    separate=restrictions.separate,
                    forbidden_routes=restrictions.forbidden_routes,
                    forced_routes=restrictions.forced_routes,
                )
            )
            continue

        assert fractional_route is not None
        stack.append(
            NodeRestrictions(
                together=restrictions.together,
                separate=restrictions.separate,
                forbidden_routes=restrictions.forbidden_routes | {fractional_route},
                forced_routes=restrictions.forced_routes,
            )
        )
        if all(
            routes[fractional_route].customer_set.isdisjoint(routes[r].customer_set)
            for r in restrictions.forced_routes
        ):
            stack.append(
                NodeRestrictions(
                    together=restrictions.together,
                    separate=restrictions.separate,
                    forbidden_routes=restrictions.forbidden_routes,
                    forced_routes=restrictions.forced_routes | {fractional_route},
                )
            )

    optimal = not stack and np.isfinite(incumbent)
    if optimal and not np.isclose(incumbent, oracle_objective, atol=1e-7):
        raise RuntimeError(
            f"branch-price-cut disagrees with full integer oracle: {incumbent} vs {oracle_objective}"
        )
    if not np.isfinite(incumbent):
        raise RuntimeError("no integer incumbent found")

    return BPCResult(
        objective=incumbent,
        selected_route_ids=incumbent_ids,
        selected_routes=tuple(routes[rid] for rid in incumbent_ids),
        nodes_processed=nodes,
        columns_generated=columns_generated,
        optimal=optimal,
        oracle_objective=oracle_objective,
    )
