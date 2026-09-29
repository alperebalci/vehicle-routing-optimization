# Routing and Vehicle-Routing Research Series

This repository belongs to a broader set of independent routing projects. The repositories are intentionally separated when the problem variant, uncertainty model, exact/heuristic method, or learning mechanism changes.

## Classical and exact routing

| Repository | Main focus | Role in the series |
|---|---|---|
| `traveling-salesman-optimization-pyomo` | Exact mathematical programming for TSP | Exact TSP foundation |
| `capacitated-vrp-branch-and-cut-python` | Exact CVRP with branch-and-cut | Exact VRP methodology |
| `exact-branch-price-cut-vrptw` | Route-based CVRPTW column generation, branching, and valid fleet cut | Exact branch-price-and-cut laboratory |
| `pickup-delivery-time-windows-ortools` | Pickup-and-delivery with time windows | Constraint/routing-solver case study |
| `pyvrp-capacitated-vehicle-routing` | Modern specialized VRP solver usage | Specialized-solver benchmark |
| `alns-vehicle-routing` | Adaptive Large Neighborhood Search for routing | Metaheuristic foundation |
| `time-dependent-vehicle-routing-alns-python` | ALNS with time-dependent travel conditions | Dynamic-cost extension |
| `multi-objective-cvrp-nsga2-python` | CVRP with multiple objectives | Multi-objective routing |

## Uncertain and dynamic routing

| Repository | Main focus | Role in the series |
|---|---|---|
| `stochastic-cvrp-sample-average-approximation-python` | CVRP under uncertainty using SAA | Stochastic routing |
| `dynamic-cvrptw-online-reoptimization-python` | Event-driven/online reoptimization for CVRPTW | Dynamic routing |
| `london-bike-share-demand-rebalancing` | Demand-aware rebalancing | Urban mobility application |
| `warehouse-multi-agent-robot-routing` | Multiple agents/robots in a shared routing environment | Multi-agent routing |
| `drone-delivery-optimization-pulp` | Delivery routing/allocation with drones | Alternative-vehicle application |

## Learning-augmented routing

| Repository | Main focus | Role in the series |
|---|---|---|
| `capacitated-vrp-rl4co-pomo-attention-model-python` | Learned constructive CVRP policy with POMO/attention | Neural routing |
| `neural-large-neighborhood-search-cvrp` | Learned guidance for neighborhood search | Neural improvement search |
| `learning-to-price-column-generation-cvrptw` | Learned pricing inside column generation | Learning-augmented exact/decomposition method |

## Why these repositories remain separate

TSP, CVRP, CVRPTW, pickup-and-delivery, stochastic VRP, dynamic VRP, multi-objective VRP, robot routing, and learned routing have different feasible regions and evaluation protocols. Likewise, branch-and-cut, ALNS, specialized VRP solvers, neural construction, neural LNS, and learned pricing intervene at fundamentally different points of the solution process.

## Suggested reading order

1. `traveling-salesman-optimization-pyomo`
2. `capacitated-vrp-branch-and-cut-python`
3. `exact-branch-price-cut-vrptw`
4. `pyvrp-capacitated-vehicle-routing`
5. `alns-vehicle-routing`
6. `time-dependent-vehicle-routing-alns-python`
7. `stochastic-cvrp-sample-average-approximation-python`
8. `dynamic-cvrptw-online-reoptimization-python`
9. `capacitated-vrp-rl4co-pomo-attention-model-python`
10. `neural-large-neighborhood-search-cvrp`
11. `learning-to-price-column-generation-cvrptw`

The ordering is pedagogical rather than a ranking of methods.