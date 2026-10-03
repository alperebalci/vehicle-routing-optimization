from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from vrp_models import (
    MINUTES_PER_DAY,
    PlanningState,
    Resource,
    RoutePlan,
    SolveResult,
    TaskPlan,
    TimeWindow,
    VRPInstance,
)

INF = np.inf

class _MILPBuilder:
    def __init__(self) -> None:
        self.c: List[float] = []
        self.lb: List[float] = []
        self.ub: List[float] = []
        self.integrality: List[int] = []
        self.names: List[str] = []
        self.rows: List[Dict[int, float]] = []
        self.row_lb: List[float] = []
        self.row_ub: List[float] = []

    def var(
        self,
        name: str,
        *,
        lb: float = 0.0,
        ub: float = INF,
        integer: bool = False,
        obj: float = 0.0,
    ) -> int:
        idx = len(self.c)
        self.c.append(float(obj))
        self.lb.append(float(lb))
        self.ub.append(float(ub))
        self.integrality.append(1 if integer else 0)
        self.names.append(name)
        return idx

    def add_obj(self, var: int, coefficient: float) -> None:
        self.c[var] += float(coefficient)

    def constraint(
        self,
        terms: Mapping[int, float],
        *,
        lb: float = -INF,
        ub: float = INF,
    ) -> None:
        clean = {int(i): float(v) for i, v in terms.items() if abs(v) > 1e-12}
        self.rows.append(clean)
        self.row_lb.append(float(lb))
        self.row_ub.append(float(ub))

    def eq(self, terms: Mapping[int, float], rhs: float) -> None:
        self.constraint(terms, lb=rhs, ub=rhs)

    def le(self, terms: Mapping[int, float], rhs: float) -> None:
        self.constraint(terms, ub=rhs)

    def ge(self, terms: Mapping[int, float], rhs: float) -> None:
        self.constraint(terms, lb=rhs)

    def solve(self, *, time_limit: Optional[float], mip_rel_gap: Optional[float]):
        n_rows = len(self.rows)
        n_cols = len(self.c)
        data: List[float] = []
        row_ind: List[int] = []
        col_ind: List[int] = []
        for r, terms in enumerate(self.rows):
            for c, v in terms.items():
                row_ind.append(r)
                col_ind.append(c)
                data.append(v)
        A = csr_matrix((data, (row_ind, col_ind)), shape=(n_rows, n_cols))
        constraints = LinearConstraint(A, np.asarray(self.row_lb), np.asarray(self.row_ub))
        options = {"presolve": True}
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
        if mip_rel_gap is not None:
            options["mip_rel_gap"] = float(mip_rel_gap)
        return milp(
            c=np.asarray(self.c),
            integrality=np.asarray(self.integrality, dtype=np.int8),
            bounds=Bounds(np.asarray(self.lb), np.asarray(self.ub)),
            constraints=constraints,
            options=options,
        )


class UnifiedWorkforceVRP:
    """Integrated multi-day workforce/vehicle routing MILP.

    The model is intended for small-to-medium educational instances. It combines
    assignment, routing and scheduling decisions in one mixed-integer model and
    supports:

    * vehicle capacity;
    * customer time windows;
    * multi-skill team coverage;
    * shift limits and paid overtime;
    * multi-day planning;
    * protected lunch breaks (depot-return policy);
    * precedence dependencies;
    * contractor fixed/variable costs;
    * priority-weighted optional visits;
    * workload fairness across in-house resources;
    * synchronized multi-resource visits;
    * rolling-horizon reoptimization through PlanningState.

    Lunch policy
    ------------
    If a shift has a lunch break, a used resource must route back to its depot by
    ``lunch_start`` and may depart again at ``lunch_start + lunch_duration``.
    This exact depot-return policy keeps the integrated routing model linear and
    prevents service or travel from being scheduled through the protected break.
    """

    START = "__START__"
    LUNCH = "__LUNCH__"
    END = "__END__"

    def __init__(self, instance: VRPInstance):
        self.instance = instance
        self.tasks = {t.id: t for t in instance.tasks}
        self.resources = {r.id: r for r in instance.resources}

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(math.hypot(a[0] - b[0], a[1] - b[1]))

    @staticmethod
    def _travel_minutes(distance_km: float, speed_kmph: float) -> float:
        return 60.0 * distance_km / speed_kmph

    def solve(
        self,
        state: Optional[PlanningState] = None,
        *,
        time_limit: Optional[float] = 30.0,
        mip_rel_gap: Optional[float] = 0.0,
    ) -> SolveResult:
        state = state or PlanningState()
        if state.current_day is not None and state.current_day >= self.instance.horizon_days:
            raise ValueError("current_day is outside the planning horizon")
        completed_at = dict(state.completed_at)
        completed_ids = set(completed_at)
        unknown_completed = completed_ids - set(self.tasks)
        if unknown_completed:
            raise ValueError(f"unknown completed tasks: {sorted(unknown_completed)}")

        active_tasks = [t for t in self.instance.tasks if t.id not in completed_ids]
        active_ids = {t.id for t in active_tasks}
        state_by_resource_day = {(s.resource_id, s.day): s for s in state.resource_day_states}
        for key, s in state_by_resource_day.items():
            if s.resource_id not in self.resources:
                raise ValueError(f"unknown resource in state: {s.resource_id}")
            if s.day not in self.resources[s.resource_id].shifts:
                raise ValueError(f"state supplied for unavailable resource-day {key}")

        locked_by_task = {l.task_id: l for l in state.locked_tasks}
        if len(locked_by_task) != len(state.locked_tasks):
            raise ValueError("duplicate locked task")
        for lock in state.locked_tasks:
            if lock.task_id not in active_ids:
                raise ValueError(f"locked task {lock.task_id} is not active")
            task = self.tasks[lock.task_id]
            if len(lock.resource_ids) != task.required_resources:
                raise ValueError(f"locked task {lock.task_id} has wrong team size")
            for rid in lock.resource_ids:
                if rid not in self.resources:
                    raise ValueError(f"unknown locked resource {rid}")

        builder = _MILPBuilder()
        horizon = self.instance.horizon_days * MINUTES_PER_DAY
        big_m = float(horizon + MINUTES_PER_DAY)

        task_start: Dict[str, int] = {}
        skip: Dict[str, int] = {}
        day_active: Dict[Tuple[str, int], int] = {}
        assign: Dict[Tuple[str, int, str], int] = {}
        use: Dict[Tuple[str, int], int] = {}
        route_end: Dict[Tuple[str, int], int] = {}
        overtime: Dict[Tuple[str, int], int] = {}
        arcs: Dict[Tuple[str, int, str, str], int] = {}
        arc_distance: Dict[Tuple[str, int, str, str], float] = {}
        arc_travel_minutes: Dict[Tuple[str, int, str, str], float] = {}
        workload: Dict[str, int] = {}

        windows_by_task_day: Dict[Tuple[str, int], TimeWindow] = {
            (t.id, w.day): w for t in active_tasks for w in t.windows
        }

        # Task-level variables.
        for task in active_tasks:
            task_start[task.id] = builder.var(
                f"start[{task.id}]", lb=0.0, ub=float(horizon)
            )
            skip_ub = 0.0 if task.mandatory else 1.0
            skip[task.id] = builder.var(
                f"skip[{task.id}]",
                lb=0.0,
                ub=skip_ub,
                integer=True,
                obj=self.instance.objective.unserved_priority_penalty * task.priority,
            )
            for day in range(self.instance.horizon_days):
                allowed = (task.id, day) in windows_by_task_day
                if state.current_day is not None and day < state.current_day:
                    allowed = False
                day_active[(task.id, day)] = builder.var(
                    f"day[{task.id},{day}]",
                    lb=0.0,
                    ub=1.0 if allowed else 0.0,
                    integer=True,
                )

            builder.eq(
                {skip[task.id]: 1.0, **{day_active[(task.id, d)]: 1.0 for d in range(self.instance.horizon_days)}},
                1.0,
            )

            for day in range(self.instance.horizon_days):
                y = day_active[(task.id, day)]
                if (task.id, day) not in windows_by_task_day:
                    continue
                w = windows_by_task_day[(task.id, day)]
                lo = day * MINUTES_PER_DAY + w.start
                hi = day * MINUTES_PER_DAY + w.end
                builder.ge({task_start[task.id]: 1.0, y: -big_m}, lo - big_m)
                builder.le({task_start[task.id]: 1.0, y: big_m}, hi + big_m)

        # Resource-day assignment, routing, shift/overtime and lunch-break variables.
        route_nodes: Dict[Tuple[str, int], List[str]] = {}
        start_info: Dict[Tuple[str, int], Tuple[float, float, int, float, bool]] = {}
        # tuple: start_x, start_y, start_abs, capacity, break_required

        for rid, resource in self.resources.items():
            for day, shift in resource.shifts.items():
                if day >= self.instance.horizon_days:
                    continue
                if state.current_day is not None and day < state.current_day:
                    continue
                key = (rid, day)
                override = state_by_resource_day.get(key)
                if override is None:
                    sx, sy = resource.depot_x, resource.depot_y
                    start_minute = shift.start
                    if state.current_day is not None and day == state.current_day:
                        start_minute = max(start_minute, int(state.current_minute))
                    capacity = resource.capacity
                    lunch_taken = False
                else:
                    sx, sy = override.x, override.y
                    start_minute = max(shift.start, override.available_minute)
                    if state.current_day is not None and day == state.current_day:
                        start_minute = max(start_minute, int(state.current_minute))
                    capacity = resource.capacity if override.remaining_capacity is None else min(resource.capacity, override.remaining_capacity)
                    lunch_taken = override.lunch_taken
                break_required = (
                    shift.lunch_start is not None
                    and not lunch_taken
                    and start_minute <= shift.lunch_start
                )
                start_abs = day * MINUTES_PER_DAY + start_minute
                start_info[key] = (sx, sy, start_abs, capacity, break_required)

                use[key] = builder.var(
                    f"use[{rid},{day}]",
                    lb=0.0,
                    ub=1.0,
                    integer=True,
                    obj=resource.fixed_day_cost,
                )
                route_end[key] = builder.var(
                    f"route_end[{rid},{day}]", lb=0.0, ub=float(horizon + shift.max_overtime)
                )
                overtime[key] = builder.var(
                    f"overtime[{rid},{day}]",
                    lb=0.0,
                    ub=float(shift.max_overtime),
                    obj=resource.overtime_cost_per_minute,
                )

                possible_tasks: List[str] = []
                for task in active_tasks:
                    allowed_day = (task.id, day) in windows_by_task_day
                    cap_ok = task.demand <= capacity + 1e-9
                    if not (allowed_day and cap_ok):
                        continue
                    v = builder.var(
                        f"assign[{rid},{day},{task.id}]",
                        lb=0.0,
                        ub=1.0,
                        integer=True,
                        obj=resource.service_cost_per_minute * task.service_minutes,
                    )
                    assign[(rid, day, task.id)] = v
                    possible_tasks.append(task.id)
                    builder.le({v: 1.0, use[key]: -1.0}, 0.0)

                if possible_tasks:
                    # A used resource-day must actually carry at least one task.
                    builder.le(
                        {use[key]: 1.0, **{assign[(rid, day, tid)]: -1.0 for tid in possible_tasks}},
                        0.0,
                    )
                else:
                    builder.le({use[key]: 1.0}, 0.0)

                # Capacity is a route-level delivery-load limit.
                if possible_tasks:
                    builder.le(
                        {assign[(rid, day, tid)]: self.tasks[tid].demand for tid in possible_tasks},
                        capacity,
                    )

                # Shift end and overtime.
                shift_end_abs = day * MINUTES_PER_DAY + shift.end
                builder.le(
                    {route_end[key]: 1.0, use[key]: -big_m},
                    0.0,
                )
                # if used: route_end <= regular end + max_overtime
                builder.le(
                    {route_end[key]: 1.0, use[key]: big_m},
                    shift_end_abs + shift.max_overtime + big_m,
                )
                # overtime >= route_end - shift_end when used
                builder.ge(
                    {overtime[key]: 1.0, route_end[key]: -1.0, use[key]: -big_m},
                    -shift_end_abs - big_m,
                )
                builder.le({overtime[key]: 1.0, use[key]: -shift.max_overtime}, 0.0)

                nodes = [self.START] + possible_tasks
                if break_required:
                    nodes.append(self.LUNCH)
                nodes.append(self.END)
                route_nodes[key] = nodes

                def coord(node: str) -> Tuple[float, float]:
                    if node == self.START:
                        return sx, sy
                    if node in (self.LUNCH, self.END):
                        return resource.depot_x, resource.depot_y
                    t = self.tasks[node]
                    return t.x, t.y

                # Directed arc set consistent with a single START->...->END path.
                arc_pairs: List[Tuple[str, str]] = []
                for i in nodes:
                    if i == self.END:
                        continue
                    for j in nodes:
                        if i == j or j == self.START:
                            continue
                        if break_required and i == self.START and j == self.END:
                            continue
                        if i == self.LUNCH and j == self.LUNCH:
                            continue
                        if j == self.LUNCH and i == self.END:
                            continue
                        if i == self.LUNCH and j == self.START:
                            continue
                        arc_pairs.append((i, j))

                for i, j in arc_pairs:
                    dist = self._distance(coord(i), coord(j))
                    tmin = self._travel_minutes(dist, resource.speed_kmph)
                    x = builder.var(
                        f"arc[{rid},{day},{i}->{j}]",
                        lb=0.0,
                        ub=1.0,
                        integer=True,
                        obj=resource.travel_cost_per_km * dist,
                    )
                    arcs[(rid, day, i, j)] = x
                    arc_distance[(rid, day, i, j)] = dist
                    arc_travel_minutes[(rid, day, i, j)] = tmin

                # Flow through start/end and mandatory lunch checkpoint.
                builder.eq(
                    {arcs[(rid, day, self.START, j)]: 1.0 for j in nodes if (rid, day, self.START, j) in arcs}
                    | {use[key]: -1.0},
                    0.0,
                )
                builder.eq(
                    {arcs[(rid, day, i, self.END)]: 1.0 for i in nodes if (rid, day, i, self.END) in arcs}
                    | {use[key]: -1.0},
                    0.0,
                )
                if break_required:
                    builder.eq(
                        {arcs[(rid, day, i, self.LUNCH)]: 1.0 for i in nodes if (rid, day, i, self.LUNCH) in arcs}
                        | {use[key]: -1.0},
                        0.0,
                    )
                    builder.eq(
                        {arcs[(rid, day, self.LUNCH, j)]: 1.0 for j in nodes if (rid, day, self.LUNCH, j) in arcs}
                        | {use[key]: -1.0},
                        0.0,
                    )

                for tid in possible_tasks:
                    incoming = {
                        arcs[(rid, day, i, tid)]: 1.0
                        for i in nodes
                        if (rid, day, i, tid) in arcs
                    }
                    outgoing = {
                        arcs[(rid, day, tid, j)]: 1.0
                        for j in nodes
                        if (rid, day, tid, j) in arcs
                    }
                    incoming[assign[(rid, day, tid)]] = -1.0
                    outgoing[assign[(rid, day, tid)]] = -1.0
                    builder.eq(incoming, 0.0)
                    builder.eq(outgoing, 0.0)

                # Temporal arc constraints. Positive service times eliminate subtours.
                lunch_start_abs = (
                    day * MINUTES_PER_DAY + shift.lunch_start
                    if break_required and shift.lunch_start is not None
                    else None
                )
                lunch_end_abs = (
                    lunch_start_abs + shift.lunch_duration
                    if lunch_start_abs is not None
                    else None
                )
                for i, j in arc_pairs:
                    x = arcs[(rid, day, i, j)]
                    travel = arc_travel_minutes[(rid, day, i, j)]
                    if i == self.START and j == self.LUNCH:
                        assert lunch_start_abs is not None
                        builder.le({x: big_m}, lunch_start_abs - start_abs - travel + big_m)
                    elif i == self.START and j == self.END:
                        builder.ge({route_end[key]: 1.0, x: -big_m}, start_abs + travel - big_m)
                    elif i == self.START:
                        builder.ge(
                            {task_start[j]: 1.0, x: -big_m},
                            start_abs + travel - big_m,
                        )
                    elif j == self.LUNCH:
                        assert lunch_start_abs is not None
                        task_i = self.tasks[i]
                        builder.le(
                            {task_start[i]: 1.0, x: big_m},
                            lunch_start_abs - task_i.service_minutes - travel + big_m,
                        )
                    elif i == self.LUNCH and j == self.END:
                        assert lunch_end_abs is not None
                        builder.ge(
                            {route_end[key]: 1.0, x: -big_m},
                            lunch_end_abs - big_m,
                        )
                    elif i == self.LUNCH:
                        assert lunch_end_abs is not None
                        builder.ge(
                            {task_start[j]: 1.0, x: -big_m},
                            lunch_end_abs + travel - big_m,
                        )
                    elif j == self.END:
                        task_i = self.tasks[i]
                        builder.ge(
                            {route_end[key]: 1.0, task_start[i]: -1.0, x: -big_m},
                            task_i.service_minutes + travel - big_m,
                        )
                    else:
                        task_i = self.tasks[i]
                        builder.ge(
                            {task_start[j]: 1.0, task_start[i]: -1.0, x: -big_m},
                            task_i.service_minutes + travel - big_m,
                        )

        # Team size and day/resource assignment linkage.
        for task in active_tasks:
            for day in range(self.instance.horizon_days):
                y = day_active[(task.id, day)]
                vars_for_day = [
                    assign[(rid, day, task.id)]
                    for rid, r in self.resources.items()
                    if (rid, day, task.id) in assign
                ]
                terms = {v: 1.0 for v in vars_for_day}
                terms[y] = -float(task.required_resources)
                builder.eq(terms, 0.0)

                for skill, required_count in task.skill_requirements.items():
                    skill_terms = {
                        assign[(rid, day, task.id)]: 1.0
                        for rid, r in self.resources.items()
                        if (rid, day, task.id) in assign and skill in r.skills
                    }
                    skill_terms[y] = -float(required_count)
                    builder.ge(skill_terms, 0.0)

        # Precedence dependencies, including dependencies on already-completed tasks.
        for task in active_tasks:
            for dep in task.predecessors:
                if dep.predecessor in completed_at:
                    builder.ge(
                        {task_start[task.id]: 1.0, skip[task.id]: big_m},
                        completed_at[dep.predecessor] + dep.min_lag,
                    )
                elif dep.predecessor in active_ids:
                    pred = self.tasks[dep.predecessor]
                    # If predecessor is skipped then successor must also be skipped.
                    builder.ge({skip[task.id]: 1.0, skip[pred.id]: -1.0}, 0.0)
                    # When successor is served, both are served and precedence is active.
                    builder.ge(
                        {
                            task_start[task.id]: 1.0,
                            task_start[pred.id]: -1.0,
                            skip[task.id]: big_m,
                        },
                        pred.service_minutes + dep.min_lag,
                    )
                else:
                    raise ValueError(
                        f"predecessor {dep.predecessor} for task {task.id} is neither active nor completed"
                    )

        # Locks pin an already-committed visit to day/start/team.
        for task_id, lock in locked_by_task.items():
            task = self.tasks[task_id]
            start_abs = lock.day * MINUTES_PER_DAY + lock.start_minute
            builder.eq({skip[task_id]: 1.0}, 0.0)
            builder.eq({day_active[(task_id, lock.day)]: 1.0}, 1.0)
            builder.eq({task_start[task_id]: 1.0}, float(start_abs))
            locked_set = set(lock.resource_ids)
            for rid in self.resources:
                v = assign.get((rid, lock.day, task_id))
                if rid in locked_set:
                    if v is None:
                        raise ValueError(f"locked assignment {rid}/{task_id} is infeasible in the model")
                    builder.eq({v: 1.0}, 1.0)
                elif v is not None:
                    builder.eq({v: 1.0}, 0.0)

        # Workload fairness across in-house resources. Workload = service + travel.
        inhouse = [r for r in self.instance.resources if not r.contractor]
        for resource in inhouse:
            w = builder.var(f"workload[{resource.id}]", lb=0.0, ub=float(horizon))
            workload[resource.id] = w
            terms: Dict[int, float] = {w: 1.0}
            for day in resource.shifts:
                for task in active_tasks:
                    v = assign.get((resource.id, day, task.id))
                    if v is not None:
                        terms[v] = terms.get(v, 0.0) - task.service_minutes
                for key, x in arcs.items():
                    rid, d, i, j = key
                    if rid == resource.id and d == day:
                        terms[x] = terms.get(x, 0.0) - arc_travel_minutes[key]
            builder.eq(terms, 0.0)

        wmax = wmin = None
        if len(inhouse) >= 2 and self.instance.objective.fairness_spread_penalty > 0:
            wmax = builder.var(
                "workload_max", lb=0.0, ub=float(horizon), obj=self.instance.objective.fairness_spread_penalty
            )
            wmin = builder.var(
                "workload_min", lb=0.0, ub=float(horizon), obj=-self.instance.objective.fairness_spread_penalty
            )
            for r in inhouse:
                builder.ge({wmax: 1.0, workload[r.id]: -1.0}, 0.0)
                builder.le({wmin: 1.0, workload[r.id]: -1.0}, 0.0)

        result = builder.solve(time_limit=time_limit, mip_rel_gap=mip_rel_gap)
        success = bool(result.x is not None and result.status in (0, 1))
        if not success:
            return SolveResult(
                success=False,
                status=int(result.status),
                message=str(result.message),
                objective_value=float(result.fun) if result.fun is not None else None,
                task_plans={},
                routes={},
                skipped_tasks=[],
                raw_result=result,
            )

        xval = np.asarray(result.x)
        task_plans: Dict[str, TaskPlan] = {}
        skipped_tasks: List[str] = []
        for task in active_tasks:
            if xval[skip[task.id]] > 0.5:
                skipped_tasks.append(task.id)
                task_plans[task.id] = TaskPlan(task.id, -1, -1, [], skipped=True)
                continue
            active_day = max(
                range(self.instance.horizon_days),
                key=lambda d: xval[day_active[(task.id, d)]],
            )
            team = sorted(
                rid
                for rid in self.resources
                if (rid, active_day, task.id) in assign and xval[assign[(rid, active_day, task.id)]] > 0.5
            )
            task_plans[task.id] = TaskPlan(
                task_id=task.id,
                day=active_day,
                start_minute=int(round(xval[task_start[task.id]] - active_day * MINUTES_PER_DAY)),
                resource_ids=team,
                skipped=False,
            )

        routes: Dict[Tuple[str, int], RoutePlan] = {}
        for (rid, day), u in use.items():
            if xval[u] <= 0.5:
                continue
            resource = self.resources[rid]
            nodes = route_nodes[(rid, day)]
            successor: Dict[str, str] = {}
            dist = 0.0
            travel_work = 0.0
            for i in nodes:
                for j in nodes:
                    key = (rid, day, i, j)
                    if key in arcs and xval[arcs[key]] > 0.5:
                        successor[i] = j
                        dist += arc_distance[key]
                        travel_work += arc_travel_minutes[key]
            seq: List[str] = []
            full_path: List[str] = [self.START]
            current = self.START
            seen = set()
            while current != self.END:
                if current in seen or current not in successor:
                    raise RuntimeError(f"could not reconstruct route for {rid} day {day}")
                seen.add(current)
                nxt = successor[current]
                full_path.append(nxt)
                if nxt not in (self.LUNCH, self.END):
                    seq.append(nxt)
                current = nxt

            last = full_path[-2]
            key_last = (rid, day, last, self.END)
            if last == self.START:
                actual_end_abs = start_info[(rid, day)][2] + arc_travel_minutes[key_last]
            elif last == self.LUNCH:
                shift = resource.shifts[day]
                actual_end_abs = day * MINUTES_PER_DAY + int(shift.lunch_start) + shift.lunch_duration
            else:
                actual_end_abs = (
                    xval[task_start[last]]
                    + self.tasks[last].service_minutes
                    + arc_travel_minutes[key_last]
                )
            shift_end_abs = day * MINUTES_PER_DAY + resource.shifts[day].end
            actual_overtime = max(0.0, float(actual_end_abs - shift_end_abs))

            service_work = sum(self.tasks[tid].service_minutes for tid in seq)
            routes[(rid, day)] = RoutePlan(
                resource_id=rid,
                day=day,
                task_ids=seq,
                end_minute=int(round(actual_end_abs - day * MINUTES_PER_DAY)),
                overtime_minutes=actual_overtime,
                distance_km=dist,
                workload_minutes=service_work + travel_work,
            )

        return SolveResult(
            success=True,
            status=int(result.status),
            message=str(result.message),
            objective_value=float(result.fun),
            task_plans=task_plans,
            routes=routes,
            skipped_tasks=sorted(skipped_tasks),
            raw_result=result,
        )


