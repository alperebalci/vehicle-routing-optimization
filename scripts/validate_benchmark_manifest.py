"""Independent manifest validation for comparative CVRP studies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate_manifest(data: dict) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if data.get("study_type") not in ("smoke", "research"):
        errors.append("study_type must be smoke or research")

    source = data.get("source")
    if not isinstance(source, dict) or not source.get("uri"):
        errors.append("source.uri is required")
    elif source.get("kind") == "external":
        digest = source.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            errors.append("external data require a verified 64-character SHA-256 digest")
    elif source.get("kind") == "synthetic":
        if type(source.get("generator_seed")) is not int:
            errors.append("synthetic data require an integer generator_seed")
    else:
        errors.append("source.kind must be external or synthetic")

    rows = data.get("runs")
    if not isinstance(rows, list) or not rows:
        return errors + ["runs must be a nonempty list"]

    grid: dict[tuple[str, int], dict[str, float]] = {}
    algorithms: set[str] = set()
    fields = ("algorithm", "instance", "seed", "time_limit_s", "runtime_s",
              "feasible", "solver_version", "hardware", "git_commit")
    for i, run in enumerate(rows):
        if not isinstance(run, dict):
            errors.append(f"runs[{i}] must be an object")
            continue
        missing = [field for field in fields if field not in run]
        if missing:
            errors.append(f"runs[{i}]: missing {', '.join(missing)}")
            continue
        for field in ("algorithm", "instance", "solver_version", "hardware", "git_commit"):
            if not isinstance(run[field], str) or not run[field].strip():
                errors.append(f"runs[{i}].{field} must be nonempty")
        if type(run["seed"]) is not int or type(run["feasible"]) is not bool:
            errors.append(f"runs[{i}]: seed must be integer and feasible must be boolean")
            continue
        for field in ("time_limit_s", "runtime_s"):
            number = run[field]
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                errors.append(f"runs[{i}].{field} must be finite and nonnegative")
        if run["feasible"]:
            value = run.get("objective")
            if type(value) not in (int, float) or not math.isfinite(value):
                errors.append(f"runs[{i}]: feasible run requires finite objective")
        if "gap_percent" in run and "reference_objective" not in run:
            errors.append(f"runs[{i}]: gap_percent requires reference_objective")
        key = (run["instance"], run["seed"])
        algo = run["algorithm"]
        if isinstance(algo, str) and isinstance(key[0], str) and type(run["time_limit_s"]) in (int, float):
            slot = grid.setdefault(key, {})
            if algo in slot:
                errors.append(f"runs[{i}]: duplicate matched result")
            slot[algo] = run["time_limit_s"]
            algorithms.add(algo)

    if data.get("study_type") == "research":
        if len(algorithms) < 2:
            errors.append("research study requires at least two algorithms")
        for key, budgets in grid.items():
            if len(budgets) != len(algorithms):
                errors.append(f"{key}: incomplete matched-algorithm grid")
            if len(set(budgets.values())) > 1:
                errors.append(f"{key}: unequal time budgets")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    issues = validate_manifest(data)
    for issue in issues:
        print(f"ERROR: {issue}")
    print(f"{len(issues)} evidence-contract issue(s)")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
