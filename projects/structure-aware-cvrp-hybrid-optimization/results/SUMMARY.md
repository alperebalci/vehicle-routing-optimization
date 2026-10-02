# Local execution results

Execution date: 2026-10-02. Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0; CPU, one requested HiGHS thread. These are synthetic short-budget experiments, not general solver rankings.

## Full versus incremental evaluation

| Experiment | Size | Paired median full/incremental time ratio |
|---|---:|---:|
| Move evaluation only | 10 | 3.36 |
| Move evaluation only | 30 | 5.62 |
| Move evaluation only | 60 | 9.79 |
| Existing root ALNS, 100 iterations | 18 | 1.60 |
| Existing root ALNS, 100 iterations | 36 | 1.29 |

Each row aggregates seeds 11, 12 and 13. Kernel checks used 5,000 identical 2-opt moves on one route. Root ALNS used the original Euclidean model and matched all six observed trajectories. These ratios must not be interpreted as speedups over an external solver.

## Fixed-time search and hybrids

Nine instances (6, 24, 60 customers; three seeds), 0.4 seconds per native run. The same audited initial solution is supplied to every method. Lower cost/initial is better.

| Variant | Mean cost / initial | Maximum overrun (seconds) |
|---|---:|---:|
| full | 0.9523 | 0.0018 |
| incremental | 0.9523 | 0.0002 |
| expanded | 0.6540 | 0.0003 |
| sequential | 0.6630 | 0.0587 |
| iterative | 0.8095 | 0.0179 |
| mip | 0.9862 | 0.0294 |

The experiment does not show a consistent advantage for the MIP hybrids over expanded local search. Late solver returns were excluded from budget-time solution improvements. In total, 81 native runs (27 fixed-proposal and 54 fixed-time) produced independently audited final solutions. Exact reference costs were available only for the six-customer cases. Larger-instance references are best observed values within this experiment, not proven optima.

Gurobi, Hexaly and PyVRP were not installed in the execution environment; no comparator run or license validation was performed. Their adapters require separate runtime validation.

The CSV contains per-run summary metrics. `summary.json` includes raw paired timings and source fingerprints. The benchmark CLI writes full input matrices, histories and MIP diagnostics to JSON on every run. The complete local JSON records are also retained in the accompanying execution bundle.
