# CVRP benchmark evidence protocol

This protocol distinguishes a passing implementation smoke test from a publishable computational comparison. The unit-test numbers in the manifest validator are illustrative, **not observed benchmark results**.

## Reproducibility requirements

1. Preserve instance identifiers, dataset licensing and origin, coordinate/rounding conventions, and a SHA-256 checksum for external data.
2. Record independent CVRP feasibility verification (each customer served exactly once, vehicle capacity, fleet limit, depot closure, and recomputed distance).
3. Compare exact/heuristic/learned baselines with the same instance, random seed and time budget, with versions, hardware, and Git revision recorded.
4. Label mathematically proved optima and best-known solutions differently. Report incumbent, lower bound and gap for exact solvers when available.
5. Run multiple random seeds and hold out final test instances from hyperparameter selection; report paired uncertainty intervals and out-of-distribution performance.
6. Record runtime and inference cost separately; smoke tests alone cannot demonstrate solver speedups.

## Machine-readable evidence gate

The root script at scripts/validate_benchmark_manifest.py validates versioned JSON manifests with schema_version=1, study_type (smoke or research), source, and runs.

Source must identify kind (external or synthetic), uri, plus SHA-256 for external or generator_seed for synthetic.

Each run needs: algorithm, instance, seed, time_limit_s, runtime_s, feasible, solver_version, hardware, git_commit, and an objective if feasible. If a gap_percent is reported, the reference_objective must be supplied and identified in the study narrative.

For research studies, each instance/seed pair must contain all algorithms on the same declared time limit. The checker does **not** independently verify that the recorded runtime or route feasibility is correct.

Run: python scripts/validate_benchmark_manifest.py path/to/manifest.json

## Next experimental stage

Integrate a checksum-pinned CVRPLIB/X instance set, validate its parser and route metric, benchmark ALNS versus PyVRP / OR-Tools and an exact method where tractable, and publish raw results with matched seeds. No external benchmark results are claimed by this infrastructure PR.
