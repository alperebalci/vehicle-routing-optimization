"""Regression tests for fair-comparison manifest validation."""

from copy import deepcopy

from scripts.validate_benchmark_manifest import validate_manifest


def fixture():
    row = {
        "algorithm": "ALNS", "instance": "synthetic-01", "seed": 7,
        "time_limit_s": 30.0, "runtime_s": 1.2, "feasible": True,
        "objective": 100.0, "solver_version": "prototype-0.1",
        "hardware": "CPU", "git_commit": "illustrative",
    }
    return {
        "schema_version": 1, "study_type": "research",
        "source": {"kind": "synthetic", "uri": "generated://test", "generator_seed": 3},
        "runs": [row, {**row, "algorithm": "greedy", "objective": 120.0}],
    }


def test_valid_matched_comparison():
    assert validate_manifest(fixture()) == []


def test_unequal_budget_rejected():
    data = fixture()
    data["runs"][1]["time_limit_s"] = 10
    assert any("unequal time budgets" in x for x in validate_manifest(data))


def test_missing_matched_instance_rejected():
    data = fixture()
    data["runs"][1]["instance"] = "another-instance"
    assert any("incomplete matched-algorithm" in x for x in validate_manifest(data))


def test_external_source_must_have_digest():
    data = fixture()
    data["source"] = {"kind": "external", "uri": "https://example.org/test.vrp"}
    assert any("SHA-256" in x for x in validate_manifest(data))


def test_no_duplicate_runs():
    data = fixture()
    data["runs"].append(deepcopy(data["runs"][0]))
    assert any("duplicate matched result" in x for x in validate_manifest(data))
