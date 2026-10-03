from __future__ import annotations

from vrp_models import (
    MINUTES_PER_DAY,
    Dependency,
    ObjectiveWeights,
    PlanningState,
    Resource,
    ResourceDayState,
    Shift,
    SolveResult,
    Task,
    TimeWindow,
    VRPInstance,
)

def minutes(hour: int, minute: int = 0) -> int:
    return 60 * hour + minute


def create_reference_instance() -> VRPInstance:
    shifts = {
        0: Shift(minutes(8), minutes(17), max_overtime=90, lunch_start=minutes(12), lunch_duration=30),
        1: Shift(minutes(8), minutes(17), max_overtime=90, lunch_start=minutes(12), lunch_duration=30),
    }
    resources = (
        Resource(
            "alice",
            0.0,
            0.0,
            capacity=12,
            skills=frozenset({"electrical", "safety"}),
            shifts=shifts,
            fixed_day_cost=40,
            travel_cost_per_km=0.8,
            service_cost_per_minute=0.35,
            overtime_cost_per_minute=0.8,
            speed_kmph=45,
        ),
        Resource(
            "bob",
            0.0,
            0.0,
            capacity=10,
            skills=frozenset({"hvac", "safety"}),
            shifts=shifts,
            fixed_day_cost=40,
            travel_cost_per_km=0.8,
            service_cost_per_minute=0.35,
            overtime_cost_per_minute=0.8,
            speed_kmph=45,
        ),
        Resource(
            "cara",
            0.0,
            0.0,
            capacity=10,
            skills=frozenset({"electrical", "hvac", "safety"}),
            shifts=shifts,
            fixed_day_cost=45,
            travel_cost_per_km=0.8,
            service_cost_per_minute=0.4,
            overtime_cost_per_minute=0.9,
            speed_kmph=45,
        ),
        Resource(
            "contractor-x",
            0.0,
            0.0,
            capacity=20,
            skills=frozenset({"electrical", "hvac", "welding", "safety"}),
            shifts=shifts,
            contractor=True,
            fixed_day_cost=220,
            travel_cost_per_km=1.4,
            service_cost_per_minute=1.2,
            overtime_cost_per_minute=2.0,
            speed_kmph=45,
        ),
    )
    tasks = (
        Task(
            "A",
            5.0,
            2.0,
            demand=4,
            service_minutes=60,
            windows=(TimeWindow(0, minutes(8, 30), minutes(10, 30)),),
            skill_requirements={"electrical": 1},
            priority=5,
            mandatory=True,
        ),
        Task(
            "B",
            8.0,
            2.0,
            demand=3,
            service_minutes=50,
            windows=(
                TimeWindow(0, minutes(10, 30), minutes(16)),
                TimeWindow(1, minutes(8, 30), minutes(12)),
            ),
            skill_requirements={"hvac": 1},
            predecessors=(Dependency("A", 20),),
            priority=4,
            mandatory=True,
        ),
        Task(
            "C",
            6.0,
            7.0,
            demand=2,
            service_minutes=75,
            windows=(
                TimeWindow(0, minutes(13), minutes(15, 30)),
                TimeWindow(1, minutes(10), minutes(15, 30)),
            ),
            required_resources=2,
            skill_requirements={"electrical": 1, "hvac": 1},
            priority=5,
            mandatory=True,
        ),
        Task(
            "D",
            2.0,
            8.0,
            demand=3,
            service_minutes=45,
            windows=(
                TimeWindow(0, minutes(14), minutes(16, 30)),
                TimeWindow(1, minutes(9), minutes(16, 30)),
            ),
            skill_requirements={"safety": 1},
            priority=2,
        ),
        Task(
            "E",
            12.0,
            4.0,
            demand=5,
            service_minutes=90,
            windows=(TimeWindow(1, minutes(9), minutes(14)),),
            skill_requirements={"electrical": 1},
            predecessors=(Dependency("B", 30),),
            priority=4,
            mandatory=True,
        ),
        Task(
            "F",
            4.0,
            12.0,
            demand=4,
            service_minutes=60,
            windows=(TimeWindow(1, minutes(11), minutes(16)),),
            skill_requirements={"welding": 1},
            priority=5,
            mandatory=True,
        ),
        Task(
            "G",
            14.0,
            10.0,
            demand=2,
            service_minutes=40,
            windows=(TimeWindow(1, minutes(13), minutes(16, 30)),),
            skill_requirements={"safety": 1},
            priority=1,
        ),
    )
    return VRPInstance(
        tasks=tasks,
        resources=resources,
        horizon_days=2,
        objective=ObjectiveWeights(
            unserved_priority_penalty=5_000.0,
            fairness_spread_penalty=1.5,
        ),
    )


def create_reoptimization_state(initial: SolveResult) -> PlanningState:
    """Example rolling-horizon state at day 0, 11:00.

    Task A is completed, each in-house vehicle reports a live location/capacity,
    and any task already committed to start before 11:30 can be locked by the
    caller. The example keeps the state deliberately simple so it can be used in
    tests and documentation.
    """

    a_plan = initial.task_plans.get("A")
    if a_plan is None or a_plan.skipped:
        raise ValueError("reference reoptimization expects task A to be served")
    completed_abs = a_plan.day * MINUTES_PER_DAY + a_plan.start_minute + 60
    return PlanningState(
        current_day=0,
        current_minute=minutes(11),
        completed_at={"A": completed_abs},
        resource_day_states=(
            ResourceDayState("alice", 0, minutes(11), 5.0, 2.0, remaining_capacity=8, lunch_taken=False),
            ResourceDayState("bob", 0, minutes(11), 0.0, 0.0, remaining_capacity=10, lunch_taken=False),
            ResourceDayState("cara", 0, minutes(11), 0.0, 0.0, remaining_capacity=10, lunch_taken=False),
            ResourceDayState("contractor-x", 0, minutes(11), 0.0, 0.0, remaining_capacity=20, lunch_taken=False),
        ),
    )
