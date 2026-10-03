from dataclasses import replace

from unified_workforce_vrp import (
    LockedTask,
    ObjectiveWeights,
    Resource,
    Shift,
    Task,
    TimeWindow,
    UnifiedWorkforceVRP,
    VRPInstance,
    create_reference_instance,
    create_reoptimization_state,
    minutes,
)


def test_reference_instance_covers_integrated_constraints():
    instance = create_reference_instance()
    result = UnifiedWorkforceVRP(instance).solve(time_limit=20)

    assert result.success, result.message
    tasks = {t.id: t for t in instance.tasks}
    resources = {r.id: r for r in instance.resources}

    for task_id, plan in result.task_plans.items():
        task = tasks[task_id]
        assert not plan.skipped
        assert len(plan.resource_ids) == task.required_resources
        window = next(w for w in task.windows if w.day == plan.day)
        assert window.start <= plan.start_minute <= window.end
        for skill, count in task.skill_requirements.items():
            qualified = sum(skill in resources[rid].skills for rid in plan.resource_ids)
            assert qualified >= count

    # Synchronized multi-resource visit with complementary skill coverage.
    assert len(result.task_plans["C"].resource_ids) == 2
    assert any("electrical" in resources[rid].skills for rid in result.task_plans["C"].resource_ids)
    assert any("hvac" in resources[rid].skills for rid in result.task_plans["C"].resource_ids)

    # Cross-task precedence with a minimum lag.
    a = result.task_plans["A"]
    b = result.task_plans["B"]
    a_abs = a.day * 1440 + a.start_minute
    b_abs = b.day * 1440 + b.start_minute
    assert b_abs >= a_abs + tasks["A"].service_minutes + 20

    # Welding exists only on the contractor in the reference data.
    assert result.task_plans["F"].resource_ids == ["contractor-x"]

    # Every route respects its vehicle capacity.
    for (rid, _day), route in result.routes.items():
        load = sum(tasks[tid].demand for tid in route.task_ids)
        assert load <= resources[rid].capacity + 1e-9


def test_overtime_lunch_and_priority_behaviour():
    # Overtime: a late mandatory visit must finish after the regular shift.
    resource = Resource(
        "r",
        0,
        0,
        capacity=10,
        skills=frozenset({"x"}),
        shifts={0: Shift(minutes(8), minutes(12), max_overtime=120)},
        overtime_cost_per_minute=1.0,
        speed_kmph=60,
    )
    late = Task(
        "late",
        10,
        0,
        demand=1,
        service_minutes=90,
        windows=(TimeWindow(0, minutes(11), minutes(12)),),
        skill_requirements={"x": 1},
        mandatory=True,
    )
    overtime_result = UnifiedWorkforceVRP(VRPInstance((late,), (resource,), 1)).solve(time_limit=5)
    assert overtime_result.success
    assert overtime_result.routes[("r", 0)].overtime_minutes >= 39.9

    # Lunch protection: this visit can be serviced before noon, but cannot be
    # finished and return to the depot by the fixed noon break, so it is dropped.
    lunch_resource = replace(
        resource,
        shifts={0: Shift(minutes(8), minutes(17), lunch_start=minutes(12), lunch_duration=30)},
    )
    lunch_conflict = Task(
        "lunch-conflict",
        30,
        0,
        demand=1,
        service_minutes=30,
        windows=(TimeWindow(0, minutes(11, 30), minutes(12)),),
        skill_requirements={"x": 1},
        priority=5,
        mandatory=False,
    )
    lunch_result = UnifiedWorkforceVRP(
        VRPInstance((lunch_conflict,), (lunch_resource,), 1)
    ).solve(time_limit=5)
    assert lunch_result.success
    assert lunch_result.task_plans["lunch-conflict"].skipped

    # Priority: with one resource and two mutually incompatible visits, preserve
    # the higher-priority task first.
    short_resource = replace(
        resource,
        shifts={0: Shift(minutes(8), minutes(10))},
        fixed_day_cost=0,
        overtime_cost_per_minute=0,
    )
    high = Task(
        "high",
        0,
        0,
        1,
        100,
        (TimeWindow(0, minutes(8), minutes(8)),),
        skill_requirements={"x": 1},
        priority=10,
    )
    low = Task(
        "low",
        0,
        0,
        1,
        100,
        (TimeWindow(0, minutes(8), minutes(8)),),
        skill_requirements={"x": 1},
        priority=1,
    )
    priority_result = UnifiedWorkforceVRP(
        VRPInstance(
            (high, low),
            (short_resource,),
            1,
            objective=ObjectiveWeights(unserved_priority_penalty=1000, fairness_spread_penalty=0),
        )
    ).solve(time_limit=5)
    assert priority_result.success
    assert not priority_result.task_plans["high"].skipped
    assert priority_result.task_plans["low"].skipped


def test_workload_fairness_splits_equal_work():
    shifts = {0: Shift(minutes(8), minutes(17))}
    r1 = Resource("r1", 0, 0, 10, frozenset({"x"}), shifts, speed_kmph=60)
    r2 = Resource("r2", 0, 0, 10, frozenset({"x"}), shifts, speed_kmph=60)
    t1 = Task(
        "t1",
        0,
        0,
        1,
        60,
        (TimeWindow(0, minutes(8), minutes(16)),),
        skill_requirements={"x": 1},
        mandatory=True,
    )
    t2 = replace(t1, id="t2")
    instance = VRPInstance(
        (t1, t2),
        (r1, r2),
        1,
        objective=ObjectiveWeights(unserved_priority_penalty=1000, fairness_spread_penalty=100),
    )
    result = UnifiedWorkforceVRP(instance).solve(time_limit=5)
    assert result.success
    assert result.task_plans["t1"].resource_ids != result.task_plans["t2"].resource_ids


def test_realtime_reoptimization_with_urgent_and_locked_visit():
    base = create_reference_instance()
    initial = UnifiedWorkforceVRP(base).solve(time_limit=20)
    assert initial.success

    state = create_reoptimization_state(initial)
    state = replace(
        state,
        locked_tasks=(LockedTask("C", 0, minutes(14), ("bob", "cara")),),
    )
    urgent = Task(
        "URGENT",
        3,
        3,
        2,
        45,
        (TimeWindow(0, minutes(13), minutes(15)),),
        skill_requirements={"electrical": 1},
        priority=10,
        mandatory=True,
    )
    updated = replace(base, tasks=base.tasks + (urgent,))
    replanned = UnifiedWorkforceVRP(updated).solve(state, time_limit=30)

    assert replanned.success, replanned.message
    assert "A" not in replanned.task_plans  # completed work is frozen outside the new model
    assert replanned.task_plans["URGENT"].day == 0
    assert replanned.task_plans["URGENT"].start_minute >= minutes(11)
    assert replanned.task_plans["C"].day == 0
    assert replanned.task_plans["C"].start_minute == minutes(14)
    assert replanned.task_plans["C"].resource_ids == ["bob", "cara"]

    for plan in replanned.task_plans.values():
        if not plan.skipped and plan.day == 0:
            assert plan.start_minute >= minutes(11)
