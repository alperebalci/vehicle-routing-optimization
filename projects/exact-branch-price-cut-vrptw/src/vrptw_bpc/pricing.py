"""Exact elementary resource-constrained label-setting pricing for small CVRPTW instances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .model import Route, VRPTWInstance


@dataclass(frozen=True)
class PricingResult:
    route: Route | None
    reduced_cost: float
    labels_generated: int


@dataclass(frozen=True)
class _Label:
    node: int
    mask: int
    time: float
    load: float
    reduced_cost: float
    path: tuple[int, ...]


def price_route_label_setting(
    instance: VRPTWInstance,
    dual_customer: ArrayLike,
    *,
    dual_vehicle: float = 0.0,
    dual_cut: float = 0.0,
    use_fleet_cut: bool = True,
    tolerance: float = 1e-12,
) -> PricingResult:
    """Solve the elementary CVRPTW pricing problem by exact label expansion.

    Arc travel cost is accumulated directly. Visiting customer j subtracts its
    set-partitioning dual. Returning to the depot adds the vehicle/cut dual terms.
    Dominance is applied only between labels with identical terminal node and visited
    set, which preserves exactness while remaining transparent for small instances.
    """

    dual = np.asarray(dual_customer, dtype=float)
    if dual.shape != (instance.n_customers,):
        raise ValueError("dual_customer has invalid shape")

    distances = instance.distances
    start_time = max(0.0, float(instance.ready[0]))
    queue: list[_Label] = [_Label(0, 0, start_time, 0.0, 0.0, ())]
    pareto: dict[tuple[int, int], list[_Label]] = {(0, 0): [queue[0]]}
    best_route: Route | None = None
    best_reduced = float("inf")
    generated = 1

    cursor = 0
    while cursor < len(queue):
        label = queue[cursor]
        cursor += 1

        if label.path:
            travel_back = float(distances[label.node, 0])
            if label.time + travel_back <= float(instance.due[0]) + tolerance:
                reduced = label.reduced_cost + travel_back - dual_vehicle
                if use_fleet_cut:
                    reduced += dual_cut
                if reduced < best_reduced - tolerance:
                    route_cost = 0.0
                    prev = 0
                    for customer in label.path:
                        route_cost += float(distances[prev, customer])
                        prev = customer
                    route_cost += float(distances[prev, 0])
                    best_route = Route(label.path, route_cost)
                    best_reduced = float(reduced)

        for customer in range(1, instance.n_customers + 1):
            bit = 1 << (customer - 1)
            if label.mask & bit:
                continue
            load = label.load + float(instance.demand[customer])
            if load > instance.capacity + tolerance:
                continue

            travel = float(distances[label.node, customer])
            arrival = label.time + travel
            service_start = max(arrival, float(instance.ready[customer]))
            if service_start > float(instance.due[customer]) + tolerance:
                continue
            departure = service_start + float(instance.service[customer])
            candidate = _Label(
                node=customer,
                mask=label.mask | bit,
                time=departure,
                load=load,
                reduced_cost=label.reduced_cost + travel - float(dual[customer - 1]),
                path=label.path + (customer,),
            )

            key = (candidate.node, candidate.mask)
            existing = pareto.get(key, [])
            if any(
                old.time <= candidate.time + tolerance
                and old.reduced_cost <= candidate.reduced_cost + tolerance
                for old in existing
            ):
                continue
            surviving = [
                old
                for old in existing
                if not (
                    candidate.time <= old.time + tolerance
                    and candidate.reduced_cost <= old.reduced_cost + tolerance
                )
            ]
            surviving.append(candidate)
            pareto[key] = surviving
            queue.append(candidate)
            generated += 1

    return PricingResult(best_route, float(best_reduced), generated)
