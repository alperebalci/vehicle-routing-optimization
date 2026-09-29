import numpy as np

from vrptw_bpc import VRPTWInstance, enumerate_routes, price_route_label_setting


def test_label_setting_matches_finite_route_pricing_oracle() -> None:
    instance = VRPTWInstance.from_arrays(
        coordinates=[[0, 0], [1, 0], [0, 1], [1, 1]],
        demand=[0, 1, 1, 1],
        ready=[0, 0, 0, 0],
        due=[20, 20, 20, 20],
        service=[0, 0, 0, 0],
        capacity=2,
        max_vehicles=2,
    )
    dual = np.array([1.4, 0.7, 1.1])
    dual_vehicle = -0.2
    dual_cut = -0.15

    result = price_route_label_setting(
        instance,
        dual,
        dual_vehicle=dual_vehicle,
        dual_cut=dual_cut,
        use_fleet_cut=True,
    )

    routes = enumerate_routes(instance)
    brute = min(
        route.cost
        - sum(dual[i - 1] for i in route.customer_set)
        - dual_vehicle
        + dual_cut
        for route in routes
    )
    assert result.route is not None
    assert np.isclose(result.reduced_cost, brute, atol=1e-10)
    assert result.labels_generated > 1
