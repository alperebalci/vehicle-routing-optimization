# Exact Branch-Price-and-Cut for Small CVRPTW Instances

A transparent research implementation of a **route-based exact CVRPTW solver** that combines restricted-master column generation, branching compatible with the route formulation, and a simple valid fleet-size cut.

The project is intentionally small. It demonstrates the mechanics and correctness contracts of branch-price-and-cut without claiming production-scale performance.

## Problem

A homogeneous fleet starts and ends at a depot. Every customer must be served exactly once, route demand must not exceed vehicle capacity, and service must begin inside each customer's time window.

The route master is

```text
minimize    sum_r c_r lambda_r
subject to  sum_r a_ir lambda_r = 1        for every customer i
            sum_r lambda_r <= K
            lambda_r binary.
```

Each column is one feasible elementary depot-to-depot route.

## Pricing design

For v0.1, the pricing oracle is deliberately finite and auditable:

1. enumerate every capacity- and time-window-feasible elementary route for a small instance;
2. retain the cheapest ordering for each customer subset;
3. start the restricted master with only a small subset of columns;
4. solve the LP and obtain customer/fleet dual values;
5. scan the still-hidden feasible routes for negative reduced cost;
6. add the most negative column and repeat.

The **master therefore uses genuine column generation**, but the pricing subproblem is exhaustive rather than a production ESPPRC label-setting algorithm. This boundary is explicit.

## Branching

When the converged route-master LP is fractional, the solver first uses a Ryan-Foster-style customer-pair disjunction:

```text
together branch: every admissible route contains both i,j or neither
separate branch: no admissible route contains both i,j.
```

With exact customer coverage, these branches enforce whether the pair is assigned to the same route.

If no fractional pair is available, the finite-universe implementation falls back to a route-variable 0/1 branch. The fallback is exact for this explicitly enumerated route universe; it is not presented as the preferred branching rule for a scalable branch-and-price solver.

## Cut

A valid fleet lower bound is added to the route master:

```text
sum_r lambda_r >= ceil(total_customer_demand / vehicle_capacity).
```

The benchmark can disable this inequality, allowing the effect on the root LP bound and search tree to be observed.

## Exactness contract

The branch-price-and-cut result is declared `optimal=True` only when the search tree is exhausted before the node limit.

For independent verification, tests also solve the **complete route-based integer master** with all enumerated columns using SciPy/HiGHS. The BPC objective must equal that oracle objective on the small benchmark instances.

Artificial high-cost coverage variables are used only to keep restricted masters feasible during column generation. A node is accepted only after all artificials are driven to zero and no negative-reduced-cost compatible route remains.

## Run

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m vrptw_bpc.experiment
```

The experiment reports:

- feasible route-universe size;
- root LP bound before and after the fleet cut;
- exact BPC objective;
- complete integer-master oracle objective;
- generated-column count;
- branch-and-bound node count;
- selected routes.

## Scope boundary

This is a solver-engineering laboratory, not a replacement for VRPSolver, SCIP, Gurobi, or specialized modern VRPTW codes.

Not yet implemented:

- ESPPRC label-setting pricing;
- ng-route relaxation;
- bidirectional labeling;
- dual stabilization;
- subset-row / rounded-capacity cuts;
- strong branching;
- column-pool management for large instances;
- benchmark-scale Solomon/Homberger experiments.

Those are the natural next steps once the finite-universe exactness harness is stable.

## References

- Desaulniers, Desrosiers, and Solomon, *Column Generation*, Springer.
- Feillet et al., exact algorithms for the elementary shortest path problem with resource constraints.
- Ryan and Foster branching for set-partitioning/column-generation formulations.

## License

This project inherits the umbrella repository's non-commercial/source-available licensing terms.
