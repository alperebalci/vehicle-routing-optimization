from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

MINUTES_PER_DAY = 24 * 60

@dataclass(frozen=True)
class TimeWindow:
    day: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.day < 0:
            raise ValueError("day must be non-negative")
        if not (0 <= self.start <= self.end <= MINUTES_PER_DAY):
            raise ValueError("time-window minutes must satisfy 0 <= start <= end <= 1440")


@dataclass(frozen=True)
class Dependency:
    predecessor: str
    min_lag: int = 0

    def __post_init__(self) -> None:
        if self.min_lag < 0:
            raise ValueError("min_lag must be non-negative")


@dataclass(frozen=True)
class Task:
    id: str
    x: float
    y: float
    demand: float
    service_minutes: int
    windows: Tuple[TimeWindow, ...]
    required_resources: int = 1
    skill_requirements: Mapping[str, int] = field(default_factory=dict)
    predecessors: Tuple[Dependency, ...] = ()
    priority: int = 1
    mandatory: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("task id must be non-empty")
        if self.demand < 0:
            raise ValueError("task demand must be non-negative")
        if self.service_minutes <= 0:
            raise ValueError("service_minutes must be positive")
        if self.required_resources <= 0:
            raise ValueError("required_resources must be positive")
        if not self.windows:
            raise ValueError("each task must have at least one time window")
        if self.priority <= 0:
            raise ValueError("priority must be positive")
        if any(v <= 0 for v in self.skill_requirements.values()):
            raise ValueError("skill requirement counts must be positive")
        if any(v > self.required_resources for v in self.skill_requirements.values()):
            raise ValueError("a skill requirement cannot exceed required_resources")
        days = [w.day for w in self.windows]
        if len(days) != len(set(days)):
            raise ValueError(f"task {self.id} has multiple windows on the same day")


@dataclass(frozen=True)
class Shift:
    start: int
    end: int
    max_overtime: int = 0
    lunch_start: Optional[int] = None
    lunch_duration: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.start < self.end <= MINUTES_PER_DAY):
            raise ValueError("shift must satisfy 0 <= start < end <= 1440")
        if self.max_overtime < 0:
            raise ValueError("max_overtime must be non-negative")
        if self.lunch_start is None:
            if self.lunch_duration != 0:
                raise ValueError("lunch_duration must be zero when lunch_start is None")
        else:
            if self.lunch_duration <= 0:
                raise ValueError("lunch_duration must be positive when lunch_start is set")
            if not (self.start <= self.lunch_start < self.end):
                raise ValueError("lunch_start must fall within the regular shift")
            if self.lunch_start + self.lunch_duration > self.end:
                raise ValueError("lunch must finish by regular shift end")


@dataclass(frozen=True)
class Resource:
    id: str
    depot_x: float
    depot_y: float
    capacity: float
    skills: frozenset[str]
    shifts: Mapping[int, Shift]
    contractor: bool = False
    fixed_day_cost: float = 0.0
    travel_cost_per_km: float = 1.0
    service_cost_per_minute: float = 0.0
    overtime_cost_per_minute: float = 0.0
    speed_kmph: float = 40.0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("resource id must be non-empty")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.speed_kmph <= 0:
            raise ValueError("speed_kmph must be positive")
        if any(c < 0 for c in (
            self.fixed_day_cost,
            self.travel_cost_per_km,
            self.service_cost_per_minute,
            self.overtime_cost_per_minute,
        )):
            raise ValueError("cost coefficients must be non-negative")


@dataclass(frozen=True)
class ResourceDayState:
    resource_id: str
    day: int
    available_minute: int
    x: float
    y: float
    remaining_capacity: Optional[float] = None
    lunch_taken: bool = False

    def __post_init__(self) -> None:
        if self.day < 0:
            raise ValueError("day must be non-negative")
        if not (0 <= self.available_minute <= MINUTES_PER_DAY):
            raise ValueError("available_minute must be in [0, 1440]")
        if self.remaining_capacity is not None and self.remaining_capacity < 0:
            raise ValueError("remaining_capacity must be non-negative")


@dataclass(frozen=True)
class LockedTask:
    task_id: str
    day: int
    start_minute: int
    resource_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.day < 0 or not (0 <= self.start_minute <= MINUTES_PER_DAY):
            raise ValueError("invalid locked-task day/start")
        if not self.resource_ids:
            raise ValueError("locked task must have at least one resource")


@dataclass(frozen=True)
class PlanningState:
    current_day: Optional[int] = None
    current_minute: Optional[int] = None
    completed_at: Mapping[str, int] = field(default_factory=dict)
    resource_day_states: Tuple[ResourceDayState, ...] = ()
    locked_tasks: Tuple[LockedTask, ...] = ()

    def __post_init__(self) -> None:
        if (self.current_day is None) != (self.current_minute is None):
            raise ValueError("current_day and current_minute must be supplied together")
        if self.current_day is not None and self.current_day < 0:
            raise ValueError("current_day must be non-negative")
        if self.current_minute is not None and not (0 <= self.current_minute <= MINUTES_PER_DAY):
            raise ValueError("current_minute must be in [0, 1440]")


@dataclass(frozen=True)
class ObjectiveWeights:
    unserved_priority_penalty: float = 10_000.0
    fairness_spread_penalty: float = 2.0

    def __post_init__(self) -> None:
        if self.unserved_priority_penalty < 0 or self.fairness_spread_penalty < 0:
            raise ValueError("objective weights must be non-negative")


@dataclass(frozen=True)
class VRPInstance:
    tasks: Tuple[Task, ...]
    resources: Tuple[Resource, ...]
    horizon_days: int
    objective: ObjectiveWeights = ObjectiveWeights()

    def __post_init__(self) -> None:
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        task_ids = [t.id for t in self.tasks]
        resource_ids = [r.id for r in self.resources]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task ids must be unique")
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("resource ids must be unique")
        task_id_set = set(task_ids)
        for task in self.tasks:
            if any(w.day >= self.horizon_days for w in task.windows):
                raise ValueError(f"task {task.id} has a window outside the planning horizon")
            for dep in task.predecessors:
                if dep.predecessor == task.id:
                    raise ValueError("task cannot depend on itself")
                if dep.predecessor not in task_id_set:
                    raise ValueError(f"unknown predecessor {dep.predecessor!r} for task {task.id}")
        for resource in self.resources:
            if any(day >= self.horizon_days or day < 0 for day in resource.shifts):
                raise ValueError(f"resource {resource.id} has a shift outside the horizon")


@dataclass
class TaskPlan:
    task_id: str
    day: int
    start_minute: int
    resource_ids: List[str]
    skipped: bool = False


@dataclass
class RoutePlan:
    resource_id: str
    day: int
    task_ids: List[str]
    end_minute: int
    overtime_minutes: float
    distance_km: float
    workload_minutes: float


@dataclass
class SolveResult:
    success: bool
    status: int
    message: str
    objective_value: Optional[float]
    task_plans: Dict[str, TaskPlan]
    routes: Dict[Tuple[str, int], RoutePlan]
    skipped_tasks: List[str]
    raw_result: object = field(repr=False, default=None)

    def summary(self) -> Dict[str, object]:
        return {
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "objective_value": self.objective_value,
            "skipped_tasks": self.skipped_tasks,
            "tasks": {
                tid: {
                    "day": p.day,
                    "start_minute": p.start_minute,
                    "resource_ids": p.resource_ids,
                    "skipped": p.skipped,
                }
                for tid, p in self.task_plans.items()
            },
            "routes": {
                f"{rid}:day{day}": {
                    "task_ids": p.task_ids,
                    "end_minute": p.end_minute,
                    "overtime_minutes": p.overtime_minutes,
                    "distance_km": p.distance_km,
                    "workload_minutes": p.workload_minutes,
                }
                for (rid, day), p in self.routes.items()
            },
        }


