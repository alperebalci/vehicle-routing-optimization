# CVRP move evaluation and MIP hybrids

This project separates three sources of search performance: candidate evaluation cost, neighborhood design, and selective use of a mixed-integer solver. It contains a route-list search implementation, a flow-based MILP, optional external adapters, and experiments with matched proposal or wall-time budgets.

## Problem contract

There is one depot (ID 0), nonnegative integer customer demands, uniform vehicle capacity, and **at most K nonempty routes**. Every customer is visited once. The sole objective is total travel cost: there is no fixed vehicle cost or lexicographic minimization of the number of vehicles.

The common input uses a symmetric, nonnegative integer cost matrix with a zero diagonal. All adapters receive that matrix unchanged. The synthetic generator uses `floor(100 * EuclideanDistance + 0.5)` once, before solving. Optional coordinates are retained for algorithms using geometric features. The PyVRP adapter requires actual coordinates rather than inventing them. Unsupported asymmetric or fractional cost data are rejected, not silently transformed.

The root ALNS ablation is a separate experiment on the existing unrounded Euclidean instances. Its results must not be combined with integer-matrix results as though they were the same instances. The older branch-and-cut subproject has an exactly-K contract and is not used unchanged as a comparator here.

## Implemented components

- `domain.py`: input validation, full route auditing, FFD/nearest-neighbor initialization, small-instance subset-DP reference. Failed initialization is not an infeasibility proof.
- `moves.py`: 2-opt, inter-route relocate, inter-route swap and tail exchange; full and incremental evaluators; route costs, loads, prefix loads and customer positions; revision-checked application and LIFO undo. Incremental candidate evaluation is constant-time for these moves. Applying accepted moves and refreshing touched routes is not constant-time.
- `mip.py`: directed arc variables, delivery flow, and separate visit flow. The latter rules out disconnected zero-demand subtours. At most K depot departures are allowed. A supplied incumbent is encoded and independently audited. SciPy/HiGHS uses an objective cutoff plus retained fallback, **not an internal MIP start**. The optional Gurobi backend supplies an actual MIP start for all arc and flow variables.
- `search.py`: matched full/incremental 2-opt search, expanded-neighborhood search, sequential local search then a full MILP, and iterative selected-route MILP reoptimization. Selected-route bounds remain labeled as neighborhood bounds; they are never promoted to global bounds.
- `adapters.py`: cold-start Gurobi configurations (`MIPFocus=0/1`), a distance-only Hexaly list model, and a PyVRP 0.14.x adapter. Missing packages and execution errors are recorded explicitly. The commercial/dedicated adapters were not executed in the recorded local experiment.

The new search kernel uses random move proposals and simulated-annealing acceptance; it is not a second ALNS implementation. The existing root ALNS retains its destroy/repair operators. Its new `local_search_evaluation` configuration only selects full or incremental 2-opt evaluation. The historical short-route and move-order rules are preserved for this ablation.

## Installation and execution

From the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pip install -e 'projects/structure-aware-cvrp-hybrid-optimization[dev]'
python -m pytest tests
python -m pytest projects/structure-aware-cvrp-hybrid-optimization/tests
```

From this project directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 cvrp-lab --mode micro \
  --sizes 10 30 60 --seeds 11 12 13 --proposals 5000 --out results/micro.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 cvrp-lab --mode alns \
  --sizes 18 36 --seeds 11 12 13 --proposals 100 --out results/alns.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 cvrp-lab --mode suite \
  --sizes 6 24 60 --seeds 11 12 13 --seconds 0.4 --proposals 3000 --out results/suite.json
cvrp-lab --mode external --instance instance.json --seconds 60 --seeds 11 12 13 \
  --out results/external.json
```

A JSON instance contains `costs`, `demands`, `capacity`, `max_vehicles`, and optionally `name` and `coordinates`. `Instance.as_dict()` exports the format. Generated instances and fingerprints are included in suite output. External data are not downloaded or redistributed by this project.

The optional extra installs Gurobi/PyVRP Python packages; it does not provide a commercial license. Hexaly installation follows its own Python API instructions. External comparisons use cold starts and are separate from the native common-incumbent ablations. Hexaly's thread setting is advisory, so a strict resource comparison additionally needs operating-system CPU isolation. No product ranking is inferred from the local results.

## Timing and evaluation

Fixed-proposal experiments compare the same proposal stream and acceptance rule. Invalid proposals are counted separately from valid evaluations. The microbenchmark measures candidate evaluation only; its ratio is not an end-to-end solver speedup.

Native wall-time budgets include initial-solution validation, cache construction, proposal generation, evaluation, accepted-move auditing, model construction, MIP calls and solution conversion. Suite input generation and the common initial solution are shared preparation and reported outside that budget. MIP calls reserve a small part of their allowance for return/audit overhead. Solvers can still overrun: overruns are measured and candidates audited after the deadline are excluded. This is not a hard real-time execution guarantee.

Histories contain observed audited improvements. MIP and external adapters currently expose the result at return, not all internal incumbent timestamps. Their time-to-target observations are consequently coarse upper bounds, not exact discovery times. The integrated gap measure excludes the initial no-solution interval, which is reported separately; it is not labeled a standard primal integral.

For n <= 11 the reference is exact subset DP. Larger examples use the best observed value in that local experiment, explicitly labeled as such, not a public best-known solution or a lower bound. Global solver bounds, neighborhood bounds, and reference gaps are separate fields.

## Recorded experiment

See `results/SUMMARY.md`, `results/summary.json` and `results/local_suite.csv`. The fixed-proposal full/incremental comparisons and all six root ALNS comparisons produced matching trajectories in the recorded runs. Short-budget hybrids did not consistently improve on expanded local search. External solver results are unavailable, not estimated.

`VALIDATION.json` records the local test scope and source fingerprints. CI runs the root regressions and the new project tests; local validation and GitHub CI status are distinct. Numerical MILP checks use explicit tolerances; there is no formal floating-point proof certificate.

## Limitations and references

The implementation targets static symmetric CVRP. It does not cover time windows, time-dependent travel, pickups, heterogeneous fleets, general typed model compilation, or arbitrary neighborhoods. The MILP is a documented flow formulation, not a full routing branch-cut-price implementation. Experiments use small synthetic instances and short budgets; neither industrial scalability nor superiority over a commercial or dedicated solver is established.

API references: [SciPy milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html), [Gurobi MIP starts](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/variable.html#start), [Hexaly CVRP modeling](https://www.hexaly.com/templates/capacitated-vehicle-routing-problem-cvrp), [PyVRP 0.14 API](https://pyvrp.readthedocs.io/en/stable/api/pyvrp.html). The vendor template's vehicle-count objective is deliberately not used because it is not part of this project's problem contract.

Repository-level license terms apply.
