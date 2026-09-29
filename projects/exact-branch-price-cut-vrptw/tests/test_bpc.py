import numpy as np

from vrptw_bpc import (
    VRPTWInstance,
    branch_price_cut,
    enumerate_routes,
    full_integer_master,
    solve_root_column_generation,
)


def triangle_instance() -> VRPTWInstance:
    return VRPTWInstance.from_arrays(
        coordinates=[
            [0.0, 0.0],
            [1.0, 0.0],
            [-0.5, 0.8660254],
            [-0.5, -0.8660254],
        ],
        demand=[0, 1, 1, 1],
        ready=[0, 0, 0, 0],
        due=[100, 100, 100, 100],
        service=[0, 0, 0, 0],
        capacity=2,
        max_vehicles=2,
    )


def test_route_enumeration_respects_capacity() -> None:
    instance = triangle_instance()
    routes = enumerate_routes(instance)

    assert len(routes) == 6
    for route in routes:
        load = np.sum(instance.demand[list(route.customers)])
        assert load <= instance.capacity + 1e-10


def test_column_generation_bound_is_valid_and_cut_is_nonweaker() -> None:
    instance = triangle_instance()
    routes = enumerate_routes(instance)
    oracle, _ = full_integer_master(instance, routes)
    without_cut = solve_root_column_generation(instance, routes, use_fleet_cut=False)
    with_cut = solve_root_column_generation(instance, routes, use_fleet_cut=True)

    assert without_cut.objective <= oracle + 1e-8
    assert with_cut.objective <= oracle + 1e-8
    assert with_cut.objective >= without_cut.objective - 1e-8


def test_branch_and_price_matches_complete_integer_master() -> None:
    instance = triangle_instance()
    result = branch_price_cut(instance, use_fleet_cut=True)

    assert result.optimal
    assert np.isclose(result.objective, result.oracle_objective, atol=1e-8)
    covered = sorted(customer for route in result.selected_routes for customer in route.customers)
    assert covered == [1, 2, 3]
    assert len(result.selected_routes) <= instance.max_vehicles


def test_branching_is_exercised_without_fleet_cut() -> None:
    instance = triangle_instance()
    result = branch_price_cut(instance, use_fleet_cut=False)

    assert result.optimal
    assert np.isclose(result.objective, result.oracle_objective, atol=1e-8)
    assert result.nodes_processed > 1


def test_time_window_can_remove_customer_pairs() -> None:
    instance = VRPTWInstance.from_arrays(
        coordinates=[[0, 0], [1, 0], [3, 0]],
        demand=[0, 1, 1],
        ready=[0, 0, 0],
        due=[100, 1.1, 3.1],
        service=[0, 0.2, 0],
        capacity=2,
        max_vehicles=2,
    )
    routes = enumerate_routes(instance)

    assert all(set(route.customers) != {1, 2} for route in routes)
