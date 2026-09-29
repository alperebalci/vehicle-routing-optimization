"""Reproducible small-instance branch-price-and-cut demonstration."""

from __future__ import annotations

import json

import numpy as np

from .model import (
    VRPTWInstance,
    branch_price_cut,
    enumerate_routes,
    solve_root_column_generation,
)


def demo_instance() -> VRPTWInstance:
    return VRPTWInstance.from_arrays(
        coordinates=[
            [0.0, 0.0],
            [1.0, 0.0],
            [-0.5, 0.8660254],
            [-0.5, -0.8660254],
            [2.0, 0.2],
            [2.0, -0.2],
        ],
        demand=[0, 1, 1, 1, 1, 1],
        ready=[0, 0, 0, 0, 0, 0],
        due=[100, 100, 100, 100, 100, 100],
        service=[0, 0, 0, 0, 0, 0],
        capacity=2,
        max_vehicles=3,
    )


def run() -> dict[str, object]:
    instance = demo_instance()
    routes = enumerate_routes(instance)
    root_without_cut = solve_root_column_generation(instance, routes, use_fleet_cut=False)
    root_with_cut = solve_root_column_generation(instance, routes, use_fleet_cut=True)
    result = branch_price_cut(instance, use_fleet_cut=True)

    return {
        "customers": instance.n_customers,
        "feasible_route_columns": len(routes),
        "root_lp_without_fleet_cut": root_without_cut.objective,
        "root_lp_with_fleet_cut": root_with_cut.objective,
        "bpc_objective": result.objective,
        "oracle_objective": result.oracle_objective,
        "nodes_processed": result.nodes_processed,
        "columns_generated": result.columns_generated,
        "optimal": result.optimal,
        "routes": [
            {"customers": list(route.customers), "cost": route.cost}
            for route in result.selected_routes
        ],
    }


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, cls=_NumpyEncoder))
