# Comprehensive Workforce Vehicle Routing MILP (SciPy/HiGHS)

An integrated multi-day **workforce routing and scheduling** model for the operational constraints that usually sit below the surface of a textbook VRP.

This project is intentionally broader than CVRP/CVRPTW. It combines routing, team assignment, workforce scheduling and rolling-horizon replanning in one mixed-integer linear model solved through `scipy.optimize.milp` (HiGHS).

## Constraint coverage

| Operational requirement | Implementation |
|---|---|
| Vehicle capacity | Route-level capacity for every resource-day |
| Time windows | Hard day-specific service-start windows |
| Skill requirements | Integer skill-coverage requirements across the assigned team |
| Shift hours | Resource/day-specific regular start and end times |
| Multi-day schedules | Tasks can have windows on multiple days; shifts are defined by day |
| Overtime | Bounded overtime with a per-minute objective cost |
| Lunch breaks | Protected fixed break with a required depot return before lunch and restart after lunch |
| Task dependencies | Precedence constraints with optional minimum lag, including cross-day dependencies |
| Contractor costs | Fixed-day, travel, service and overtime costs by resource; contractors are first-class resources |
| Priority visits | Optional-task skip penalty is weighted by task priority; mandatory tasks cannot be skipped |
| Workload fairness | Max-minus-min workload penalty across in-house resources |
| Multi-resource visits | Exact team-size constraints plus synchronized start time across all assigned resources |
| Real-time rescheduling | `PlanningState` supports completed tasks, live resource location/time/capacity and locked commitments |

The model is best interpreted as a **multi-day multi-skill workforce/technician routing problem with synchronized visits and dynamic reoptimization** rather than a plain VRP.

## Modeling choices

### Routing

Each used resource-day is represented by a single path from its live/start location to its depot. Task nodes use standard in/out flow conservation. Positive service times plus temporal arc constraints eliminate disconnected task subtours.

Travel is Euclidean and converted to minutes using each resource's speed. The model can be extended to a matrix/API travel-time source by replacing the distance/travel functions.

### Lunch break

A used resource with a configured lunch break must return to its depot by `lunch_start`. It may depart again at `lunch_start + lunch_duration`.

This is deliberately stronger than a generic "break anywhere" rule, but it has two useful properties:

1. service and travel cannot silently pass through lunch;
2. the integrated routing/scheduling formulation stays linear and easy to audit.

For field operations where lunch may be taken at the current customer/location, the lunch checkpoint can be generalized into a location-dependent break formulation.

### Multi-resource visits and skills

A task may require `required_resources > 1`. All assigned resources share the same task start variable, so service is synchronized automatically.

Skill requirements are counts, not simple labels. For example:

```python
required_resources=2
skill_requirements={"electrical": 1, "hvac": 1}
```

requires a two-person synchronized team with at least one electrical-qualified and one HVAC-qualified resource. One multi-skilled person may satisfy both coverage inequalities, but the team-size requirement still forces two resources.

### Priority and optional work

Every non-mandatory task has a binary skip variable. Skipping incurs

```text
unserved_priority_penalty * priority
```

so high-priority visits are preserved before low-priority work when the problem becomes infeasible or uneconomic.

### Fairness

For in-house resources, workload is

```text
service minutes + travel minutes
```

and the objective penalizes `max(workload) - min(workload)`. Contractors are excluded from this fairness term.

### Real-time replanning

`PlanningState` changes the planning origin without rebuilding application logic:

- `current_day` / `current_minute` prevent scheduling into the past;
- `completed_at` removes completed work while preserving precedence timing;
- `ResourceDayState` updates current location, next-available time and remaining vehicle capacity;
- `LockedTask` pins a committed visit to its day, start time and assigned team.

This gives a rolling-horizon reoptimization workflow suitable for traffic events, urgent jobs, delays, absences or customer changes.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.10+ is supported.

## Run the reference scenario

```bash
python example.py
```

The reference instance contains:

- two planning days;
- three in-house resources plus one expensive contractor;
- capacity constraints;
- electrical/HVAC/safety/welding skills;
- a synchronized two-resource visit;
- precedence constraints;
- protected lunch returns;
- overtime limits;
- priority-weighted optional work;
- a contractor-only welding job;
- fairness in the objective.

The example then advances to day 0 at 11:00, marks task `A` complete, injects an urgent task, reports live resource states, locks a synchronized visit, and resolves the remaining horizon.

## Minimal API

```python
from unified_workforce_vrp import UnifiedWorkforceVRP, create_reference_instance

instance = create_reference_instance()
result = UnifiedWorkforceVRP(instance).solve(time_limit=30)

if result.success:
    print(result.summary())
```

A task with multiple day choices can be created as:

```python
Task(
    id="visit-17",
    x=8.0,
    y=3.0,
    demand=4,
    service_minutes=75,
    windows=(
        TimeWindow(day=0, start=9 * 60, end=15 * 60),
        TimeWindow(day=1, start=10 * 60, end=16 * 60),
    ),
    required_resources=2,
    skill_requirements={"electrical": 1, "hvac": 1},
    predecessors=(Dependency("visit-09", min_lag=30),),
    priority=5,
    mandatory=True,
)
```

## Tests

```bash
python -m pytest -q
```

The tests cover the integrated reference solution, overtime, lunch protection, priority-based dropping, workload fairness and rolling-horizon replanning with a locked multi-resource visit.

## Scope and limitations

This is an auditable research/portfolio implementation, not a production dispatcher.

- The arc-flow MILP grows roughly as `resources × days × tasks²`; decomposition or ALNS/column generation is preferable for large fleets.
- Travel is deterministic and Euclidean.
- Lunch is a depot-return policy rather than a flexible break-location model.
- Capacity is a route-level delivery-load constraint; pickup-and-delivery inventory propagation is not included.
- The model does not include stochastic travel, robust optimization, charging, heterogeneous commodities or split deliveries.
- HiGHS can return a feasible incumbent at a time limit; inspect the solver status and MIP gap for serious computational studies.

## Why this project exists

The repository already contains focused implementations for CVRP, CVRPTW, time-dependent routing and online reoptimization. Those are useful isolated models, but real field-service routing typically requires the routing and workforce constraints above simultaneously. This project is the integrated version.
