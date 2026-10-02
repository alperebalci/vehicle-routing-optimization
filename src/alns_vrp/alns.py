from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .construct import greedy_initial_solution
from .data import CVRPInstance
from .local_search import two_opt
from .operators import AdaptivePool, DESTROY_OPERATORS, REPAIR_OPERATORS
from .solution import Solution


@dataclass(frozen=True)
class ALNSConfig:
    iterations: int = 3000
    min_removal: int = 2
    max_removal_fraction: float = 0.30
    segment_length: int = 100
    reaction: float = 0.20
    cooling_rate: float = 0.997
    initial_worsening_fraction: float = 0.05
    initial_acceptance_probability: float = 0.50
    score_best: float = 8.0
    score_improve: float = 4.0
    score_accept: float = 1.0
    local_search: bool = True
    local_search_evaluation: str = "incremental"

    def validate(self, n_customers: int) -> None:
        if self.local_search_evaluation not in {"full", "incremental"}:
            raise ValueError("local_search_evaluation must be full or incremental")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.min_removal <= 0:
            raise ValueError("min_removal must be positive")
        if not 0 < self.max_removal_fraction <= 1:
            raise ValueError("max_removal_fraction must be in (0, 1]")
        if self.segment_length <= 0:
            raise ValueError("segment_length must be positive")
        if not 0 < self.reaction <= 1:
            raise ValueError("reaction must be in (0, 1]")
        if not 0 < self.cooling_rate <= 1:
            raise ValueError("cooling_rate must be in (0, 1]")
        if self.initial_worsening_fraction <= 0:
            raise ValueError("initial_worsening_fraction must be positive")
        if not 0 < self.initial_acceptance_probability < 1:
            raise ValueError("initial_acceptance_probability must be in (0, 1)")
        if self.min_removal > n_customers:
            raise ValueError("min_removal cannot exceed number of customers")


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    current_cost: float
    best_cost: float
    temperature: float
    destroy_operator: str
    repair_operator: str
    accepted: bool


@dataclass(frozen=True)
class ALNSResult:
    best_solution: Solution
    best_cost: float
    initial_cost: float
    records: tuple[IterationRecord, ...]
    destroy_weights: dict[str, float]
    repair_weights: dict[str, float]

    @property
    def improvement_percent(self) -> float:
        return 100.0 * (self.initial_cost - self.best_cost) / self.initial_cost


def _initial_temperature(initial_cost: float, config: ALNSConfig) -> float:
    worsening = max(initial_cost * config.initial_worsening_fraction, 1e-9)
    return -worsening / math.log(config.initial_acceptance_probability)


def solve_alns(
    instance: CVRPInstance,
    config: ALNSConfig | None = None,
    seed: int = 2026,
    initial_solution: Solution | None = None,
) -> ALNSResult:
    cfg = config or ALNSConfig()
    cfg.validate(instance.n_customers)
    rng = random.Random(seed)

    current = (initial_solution.copy() if initial_solution is not None else greedy_initial_solution(instance)).normalized()
    if not current.is_feasible(instance):
        raise ValueError("initial_solution must be complete and feasible")
    if cfg.local_search:
        current = two_opt(current, instance, evaluation=cfg.local_search_evaluation)

    current_cost = current.total_distance(instance)
    initial_cost = current_cost
    best = current.copy()
    best_cost = current_cost
    temperature = _initial_temperature(initial_cost, cfg)

    destroy_pool = AdaptivePool(list(DESTROY_OPERATORS), cfg.reaction)
    repair_pool = AdaptivePool(list(REPAIR_OPERATORS), cfg.reaction)
    records: list[IterationRecord] = []

    max_removal = max(cfg.min_removal, int(math.ceil(cfg.max_removal_fraction * instance.n_customers)))
    max_removal = min(max_removal, max(1, instance.n_customers - 1))
    min_removal = min(cfg.min_removal, max_removal)

    for iteration in range(1, cfg.iterations + 1):
        destroy_name = destroy_pool.choose(rng)
        repair_name = repair_pool.choose(rng)
        q = rng.randint(min_removal, max_removal)

        partial, removed = DESTROY_OPERATORS[destroy_name](instance, current, q, rng)
        candidate = REPAIR_OPERATORS[repair_name](instance, partial, removed, rng)
        if cfg.local_search:
            candidate = two_opt(candidate, instance, evaluation=cfg.local_search_evaluation)
        if not candidate.is_feasible(instance):
            raise RuntimeError("operator pair generated an infeasible solution")

        candidate_cost = candidate.total_distance(instance)
        delta = candidate_cost - current_cost
        accepted = delta <= 1e-12 or rng.random() < math.exp(-delta / max(temperature, 1e-12))

        reward = 0.0
        if candidate_cost < best_cost - 1e-9:
            best = candidate.copy()
            best_cost = candidate_cost
            reward = cfg.score_best
        elif candidate_cost < current_cost - 1e-9:
            reward = cfg.score_improve
        elif accepted:
            reward = cfg.score_accept

        if reward > 0:
            destroy_pool.reward(destroy_name, reward)
            repair_pool.reward(repair_name, reward)

        if accepted:
            current = candidate
            current_cost = candidate_cost

        records.append(
            IterationRecord(
                iteration=iteration,
                current_cost=current_cost,
                best_cost=best_cost,
                temperature=temperature,
                destroy_operator=destroy_name,
                repair_operator=repair_name,
                accepted=accepted,
            )
        )

        if iteration % cfg.segment_length == 0:
            destroy_pool.update()
            repair_pool.update()
        temperature *= cfg.cooling_rate

    if cfg.iterations % cfg.segment_length != 0:
        destroy_pool.update()
        repair_pool.update()

    return ALNSResult(
        best_solution=best.normalized(),
        best_cost=best_cost,
        initial_cost=initial_cost,
        records=tuple(records),
        destroy_weights=destroy_pool.weights(),
        repair_weights=repair_pool.weights(),
    )
