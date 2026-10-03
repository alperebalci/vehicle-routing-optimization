from __future__ import annotations

import json
from dataclasses import replace

from unified_workforce_vrp import (
    LockedTask,
    Task,
    TimeWindow,
    UnifiedWorkforceVRP,
    create_reference_instance,
    create_reoptimization_state,
    minutes,
)


def main() -> None:
    base = create_reference_instance()
    initial = UnifiedWorkforceVRP(base).solve(time_limit=30)
    if not initial.success:
        raise RuntimeError(initial.message)

    print("INITIAL PLAN")
    print(json.dumps(initial.summary(), indent=2))

    state = create_reoptimization_state(initial)
    state = replace(
        state,
        locked_tasks=(
            LockedTask(
                task_id="C",
                day=0,
                start_minute=minutes(14),
                resource_ids=("bob", "cara"),
            ),
        ),
    )

    urgent = Task(
        id="URGENT",
        x=3.0,
        y=3.0,
        demand=2,
        service_minutes=45,
        windows=(TimeWindow(0, minutes(13), minutes(15)),),
        skill_requirements={"electrical": 1},
        priority=10,
        mandatory=True,
    )
    updated = replace(base, tasks=base.tasks + (urgent,))

    replanned = UnifiedWorkforceVRP(updated).solve(state, time_limit=30)
    if not replanned.success:
        raise RuntimeError(replanned.message)

    print("\nREPLANNED AT DAY 0 11:00")
    print(json.dumps(replanned.summary(), indent=2))


if __name__ == "__main__":
    main()
