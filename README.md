# Vehicle Routing Optimization

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is the primary umbrella repository for this Jors Academy research area. Related projects have been consolidated under `projects/` so the methods, implementations, experiments, and case studies can be maintained and explored from one place.

### Included projects

- [`capacitated-vrp-branch-and-cut-python`](projects/capacitated-vrp-branch-and-cut-python/)
- [`dynamic-cvrptw-online-reoptimization-python`](projects/dynamic-cvrptw-online-reoptimization-python/)
- [`exact-branch-price-cut-vrptw`](projects/exact-branch-price-cut-vrptw/)
- [`multi-objective-cvrp-nsga2-python`](projects/multi-objective-cvrp-nsga2-python/)
- [`pickup-delivery-time-windows-ortools`](projects/pickup-delivery-time-windows-ortools/)
- [`pyvrp-capacitated-vehicle-routing`](projects/pyvrp-capacitated-vehicle-routing/)
- [`rl-hyper-heuristic-cvrp`](projects/rl-hyper-heuristic-cvrp/)
- [`stochastic-cvrp-sample-average-approximation-python`](projects/stochastic-cvrp-sample-average-approximation-python/)
- [`time-dependent-vehicle-routing-alns-python`](projects/time-dependent-vehicle-routing-alns-python/)
- [`traveling-salesman-optimization-pyomo`](projects/traveling-salesman-optimization-pyomo/)

Each consolidated project keeps its own files and a `SOURCE_REPOSITORY.md` provenance record. The snapshot preserves the source repository's default-branch files at consolidation time; repository-level history and metadata remain separate from the snapshot. New projects developed natively inside this umbrella repository are listed alongside those consolidated snapshots.
<!-- portfolio-umbrella:end -->

A reproducible Operations Research implementation of **Adaptive Large Neighborhood Search (ALNS)** for the capacitated vehicle routing problem (CVRP). The repository is designed to show the mechanics of a modern problem-specific metaheuristic rather than hide the search behind a library call.

The algorithm repeatedly destroys part of a feasible routing plan, repairs it with a constructive heuristic, optionally improves routes with 2-opt, and decides whether to accept the new solution using a simulated-annealing criterion. Destroy and repair operators are selected adaptively according to their observed search performance.

## Problem

A homogeneous fleet starts and ends at one depot. Every customer must be visited exactly once, vehicle capacity may not be exceeded, and at most `K` vehicles may be used.

For route set `R`, the objective is

```text
minimize    total Euclidean travel distance

subject to  every customer is visited exactly once
            route demand <= vehicle capacity
            number of nonempty routes <= K
```

The demo data are synthetic Euclidean CVRP instances. This repository is methodological; it is not a calibrated routing model for a specific company.

## Why ALNS?

Large Neighborhood Search changes many decisions at once instead of relying only on small local moves. ALNS extends this idea by maintaining multiple destroy and repair heuristics and changing their selection probabilities according to historical performance.

Ropke and Pisinger's 2006 work introduced the adaptive framework for pickup-and-delivery with time windows. The core idea is directly transferable to CVRP: competing subheuristics are selected with frequencies reflecting their past success. The implementation here adapts that framework to a simpler capacitated-routing setting.

Every ALNS iteration performs

```text
current solution
      |
      v
select destroy operator using adaptive weights
      |
      v
remove q customers
      |
      v
select repair operator using adaptive weights
      |
      v
reinsert all removed customers feasibly
      |
      v
2-opt route improvement
      |
      v
simulated-annealing acceptance
      |
      v
update incumbent and operator scores
```

## Implemented destroy operators

### Random removal

Selects `q` customers uniformly without replacement. It is primarily a diversification operator.

### Worst removal

Ranks customers by their marginal removal saving

```text
c(prev, i) + c(i, next) - c(prev, next)
```

and removes customers that currently contribute strongly to route cost.

### Related removal

A Shaw-style relatedness operator chooses a seed customer and preferentially removes customers that are close in space and similar in demand. The intent is to remove a coherent region of the incumbent solution so that repair can rebuild it differently.

## Implemented repair operators

### Greedy insertion

For every unassigned customer, all capacity-feasible insertion positions are evaluated. The globally cheapest insertion is performed repeatedly until the solution is complete.

### Regret-2 insertion

For each unassigned customer, the two best insertion costs are computed. The customer with the largest difference between its second-best and best options is inserted first. This prioritizes customers for which postponing insertion is expensive.

Greedy and regret insertion are standard repair ideas in the ALNS literature.

## Adaptive operator selection

Destroy and repair operators begin with equal weights. Selection uses roulette-wheel sampling proportional to the current weights.

During each segment, selected operators receive rewards when their candidate solution

- creates a new global best,
- improves the current solution, or
- is accepted despite not improving it.

At the end of a segment, an operator's weight is blended with its average segment score:

```text
new_weight = (1 - reaction) * old_weight
             + reaction * segment_performance
```

This is intentionally transparent. The exact scoring parameters are configuration values rather than claims of universal optimality.

## Acceptance criterion

A better candidate is always accepted. A worse candidate with objective increase `Delta` is accepted with probability

```text
exp(-Delta / T)
```

where `T` decreases geometrically. The initial temperature is calibrated from a requested worsening fraction and acceptance probability. This provides controlled diversification early in the search and increasingly greedy behavior later.

## Local search

After repair, deterministic best-improvement **2-opt** is applied independently to each route. The destroy/repair layer changes customer assignment and route structure; 2-opt removes avoidable crossings and improves within-route ordering.

## Exact small-instance benchmark

`src/alns_vrp/exact.py` contains a subset-dynamic-programming solver for small CVRP instances. It combines

1. a Held-Karp-style dynamic program to find the optimal depot tour for every capacity-feasible customer subset, and
2. a set-partition dynamic program to split all customers into at most `K` feasible routes.

It is intentionally limited to small instances because its exponential complexity is unsuitable for realistic CVRP sizes. Its purpose is verification: unit tests include a known instance where both the exact solver and ALNS obtain the optimum value `8.0`.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── examples/run_demo.py
├── src/alns_vrp/
│   ├── __init__.py
│   ├── __main__.py
│   ├── alns.py
│   ├── construct.py
│   ├── data.py
│   ├── exact.py
│   ├── experiment.py
│   ├── local_search.py
│   ├── operators.py
│   └── solution.py
├── tests/
│   ├── test_alns.py
│   ├── test_data_solution.py
│   ├── test_local_search_cli.py
│   └── test_operators.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.10+ is supported. The solver itself uses only the Python standard library; `pytest` is the development dependency.

## Run

```bash
python -m alns_vrp \
  --iterations 3000 \
  --customers 36 \
  --seed 2026
```

or

```bash
python examples/run_demo.py
```

The JSON output contains

- the initial and best route distances,
- percentage improvement,
- the selected routes,
- route loads,
- final destroy-operator weights,
- final repair-operator weights,
- the final iteration state.

With the repository's deterministic demo seed, a 1000-iteration run reduced the greedy/2-opt starting distance from about `812.98` to about `680.72`, a reduction of about `16.27%`. This is a reproducible demonstration result, not a benchmark claim against best-known CVRP solutions.

## Tests

```bash
python -m pytest
```

The suite verifies

- CVRP data validation,
- deterministic synthetic-instance generation,
- duplicate-customer detection,
- feasibility restoration for every destroy/repair operator pair,
- adaptive weight updates,
- 2-opt non-worsening behavior,
- exact small-instance optimization,
- exact/ALNS agreement on a known optimum,
- ALNS reproducibility under a fixed seed,
- monotone best-so-far objective history,
- complete capacity-feasible solutions,
- valid CLI JSON output.

GitHub Actions installs the package, compiles the source tree, and runs the complete suite on Python 3.10 and 3.12.

## Methodological notes

- ALNS is a metaheuristic. A good result on a large instance is not a proof of global optimality.
- Adaptive weights change operator-selection probabilities; they do not learn a value function or policy in the reinforcement-learning sense.
- Simulated annealing is used only as the candidate acceptance mechanism. It is not the neighborhood generator.
- Related removal in this repository is inspired by Shaw-style relatedness, but the exact relatedness formula is intentionally simplified for CVRP.
- The exact dynamic program is a verification tool for small instances, not the production solver.
- Serious computational studies should use established benchmark sets, multiple random seeds, equal time/evaluation budgets, and statistical comparisons against strong baselines.

## References

- S. Ropke and D. Pisinger, *An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows*, Transportation Science 40(4), 455-472, 2006. https://doi.org/10.1287/trsc.1050.0135
- P. Shaw, *Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems*, CP 1998, Lecture Notes in Computer Science 1520, 417-431. https://doi.org/10.1007/3-540-49481-2_30
- D. Pisinger and S. Ropke, *A General Heuristic for Vehicle Routing Problems*, Computers & Operations Research 34(8), 2403-2435, 2007. https://doi.org/10.1016/j.cor.2005.09.012

## License

This repository is licensed under the **JORS Academy Non-Commercial Source License 1.0**. Commercial use is prohibited without a separate prior written commercial license. See [`LICENSE`](LICENSE) for the complete terms.
