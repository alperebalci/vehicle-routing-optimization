from __future__ import annotations

from .data import CVRPInstance
from .solution import Solution, route_distance


def two_opt(
    solution: Solution, instance: CVRPInstance, *, evaluation: str = "incremental"
) -> Solution:
    """Best-improvement 2-opt for the symmetric Euclidean CVRPInstance.

    ``full`` retains the original evaluator for controlled comparisons.
    Both modes use the original move order, tolerance and short-route rule.
    Delta evaluation is O(1); applying an accepted reversal remains O(m).
    The four-edge formula must not be reused for asymmetric travel costs.
    """
    if evaluation not in {"full", "incremental"}:
        raise ValueError("evaluation must be 'full' or 'incremental'")
    improved = solution.copy()
    for route_index, route in enumerate(improved.routes):
        if len(route) < 4:
            continue
        while True:
            base = route_distance(instance, route)
            best_delta = 0.0
            best_move: tuple[int, int] | None = None
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route) + 1):
                    if i == 0 and j == len(route):
                        continue
                    if evaluation == "full":
                        candidate = route[:i] + list(reversed(route[i:j])) + route[j:]
                        delta = route_distance(instance, candidate) - base
                    else:
                        a = 0 if i == 0 else route[i - 1]
                        b, c = route[i], route[j - 1]
                        d = 0 if j == len(route) else route[j]
                        delta = (
                            instance.distance(a, c) + instance.distance(b, d)
                            - instance.distance(a, b) - instance.distance(c, d)
                        )
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_move = (i, j)
            if best_move is None:
                break
            i, j = best_move
            route = route[:i] + list(reversed(route[i:j])) + route[j:]
            improved.routes[route_index] = route
    return improved
