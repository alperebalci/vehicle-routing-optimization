"""Common symmetric CVRP contract and independent solution auditing."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Instance:
    costs: np.ndarray
    demands: np.ndarray
    capacity: int
    max_vehicles: int
    name: str = "instance"
    coordinates: np.ndarray | None = None

    def __post_init__(self):
        c, q = np.asarray(self.costs), np.asarray(self.demands)
        if c.ndim != 2 or c.shape[0] < 2 or c.shape[0] != c.shape[1]:
            raise ValueError("costs must be a square matrix including depot 0")
        if q.shape != (len(c),):
            raise ValueError("demand shape mismatch")
        for arr in (c, q):
            if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
                raise ValueError("finite numeric data required")
            if np.any(arr < 0) or np.any(arr > 10**9) or np.any(arr != np.floor(arr)):
                raise ValueError("nonnegative integral values <= 1e9 required")
        if not np.array_equal(c, c.T) or np.any(np.diag(c) != 0):
            raise ValueError("symmetric costs and zero diagonal required")
        if q[0] != 0:
            raise ValueError("depot demand must be zero")
        for v in (self.capacity, self.max_vehicles):
            if isinstance(v, bool) or not isinstance(v, Integral) or v <= 0:
                raise ValueError("capacity and max_vehicles must be positive integers")
        if self.capacity > 10**9 or np.any(q > self.capacity):
            raise ValueError("invalid capacity or individual demand")
        if self.coordinates is not None:
            xy = np.array(self.coordinates, dtype=float, copy=True)
            if xy.shape != (len(c), 2) or not np.isfinite(xy).all():
                raise ValueError("coordinates must have finite shape (n+1, 2)")
            xy.setflags(write=False)
            object.__setattr__(self, "coordinates", xy)
        c, q = c.astype(np.int64, copy=True), q.astype(np.int64, copy=True)
        c.setflags(write=False)
        q.setflags(write=False)
        object.__setattr__(self, "costs", c)
        object.__setattr__(self, "demands", q)
        object.__setattr__(self, "max_vehicles", min(self.max_vehicles, self.n))

    @property
    def n(self):
        return len(self.demands) - 1

    def as_dict(self):
        return {"name": self.name, "costs": self.costs.tolist(),
                "demands": self.demands.tolist(), "capacity": int(self.capacity),
                "max_vehicles": int(self.max_vehicles),
                "coordinates": self.coordinates.tolist() if self.coordinates is not None else None}

    def fingerprint(self):
        data = self.as_dict()
        data.pop("name")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def route_cost(p: Instance, route) -> int:
    """Full recomputation; deliberately independent of all move caches."""
    points = (0, *route, 0)
    return sum(int(p.costs[a, b]) for a, b in zip(points, points[1:]))


def audit(p: Instance, routes) -> int:
    flat = [v for route in routes for v in route]
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in flat):
        raise ValueError("customer IDs must be integers")
    if sorted(flat) != list(range(1, p.n + 1)):
        raise ValueError("every customer must appear exactly once")
    if sum(bool(route) for route in routes) > p.max_vehicles:
        raise ValueError("at most K nonempty routes allowed")
    if any(sum(int(p.demands[v]) for v in route) > p.capacity for route in routes):
        raise ValueError("capacity violation")
    return sum(route_cost(p, route) for route in routes)


def initial_routes(p: Instance):
    """FFD packing, then nearest-neighbor ordering; failure is NOT infeasibility."""
    bins, loads = [], []
    for v in sorted(range(1, p.n + 1), key=lambda x: (-int(p.demands[x]), x)):
        k = next((i for i, load in enumerate(loads)
                  if load + p.demands[v] <= p.capacity), len(bins))
        if k == len(bins):
            if k == p.max_vehicles:
                raise RuntimeError("FFD initialization failed; provide a feasible initial solution")
            bins.append([])
            loads.append(0)
        bins[k].append(v)
        loads[k] += int(p.demands[v])
    routes = []
    for customers in bins:
        route, left, prev = [], set(customers), 0
        while left:
            v = min(left, key=lambda j: (p.costs[prev, j], j))
            route.append(v)
            left.remove(v)
            prev = v
        routes.append(route)
    audit(p, routes)
    return routes


def generated(seed=0, n=30, clustered=False):
    if not isinstance(n, int) or n < 2:
        raise ValueError("n must be at least two")
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, (n + 1, 2))
    if clustered:
        centers = np.array([[20, 20], [80, 80], [20, 80], [80, 20]])
        xy[1:] = centers[np.arange(n) % 4] + rng.normal(0, 6, (n, 2))
    xy[0] = 50
    d = np.floor(np.linalg.norm(xy[:, None] - xy[None, :], axis=2) * 100 + .5)
    q = np.r_[0, rng.integers(1, 7, n)]
    cap = 24
    # Derive K from a known feasible packing, not from a claimed sufficiency of total capacity.
    probe = Instance(d, q, cap, n)
    k = min(n, len(initial_routes(probe)) + 1)
    return Instance(d, q, cap, k, f"{'clustered' if clustered else 'uniform'}-{n}-{seed}", xy)


def load(path):
    obj = json.loads(Path(path).read_text())
    if set(obj) - {"costs", "demands", "capacity", "max_vehicles", "name", "coordinates"}:
        raise ValueError("unknown input fields")
    return Instance(**obj)


def exact_cost(p: Instance, limit=11):
    """Held-Karp subset tours and set-partition DP; at most K routes."""
    if p.n > limit:
        raise ValueError("exact reference size limit exceeded")
    n, size = p.n, 1 << p.n
    load = np.zeros(size, dtype=np.int64)
    tsp = np.full((size, n), np.inf)
    costs = np.full(size, np.inf)
    costs[0] = 0
    for mask in range(1, size):
        bit = mask & -mask
        load[mask] = load[mask ^ bit] + p.demands[bit.bit_length()]
        if load[mask] > p.capacity:
            continue
        for j in range(n):
            if not mask & (1 << j):
                continue
            rest = mask ^ (1 << j)
            tsp[mask, j] = (p.costs[0, j + 1] if rest == 0 else
                           min(tsp[rest, k] + p.costs[k + 1, j + 1]
                               for k in range(n) if rest & (1 << k)))
        costs[mask] = min(tsp[mask, j] + p.costs[j + 1, 0]
                          for j in range(n) if mask & (1 << j))

    @lru_cache(None)
    def partition(mask, k):
        if not mask:
            return 0.0
        if not k:
            return math.inf
        anchor, sub, best = mask & -mask, mask, math.inf
        while sub:
            if sub & anchor and np.isfinite(costs[sub]):
                best = min(best, costs[sub] + partition(mask ^ sub, k - 1))
            sub = (sub - 1) & mask
        return best
    return float(partition(size - 1, p.max_vehicles))
