"""Route moves with O(1) deltas, touched-route caches, and transactional undo."""
from __future__ import annotations

from dataclasses import dataclass
import random

from .domain import Instance, audit, route_cost


@dataclass(frozen=True)
class Move:
    kind: str
    r: int
    i: int
    s: int
    j: int


@dataclass(frozen=True)
class Evaluation:
    move: Move
    delta: int
    revision: int
    owner: object


@dataclass(frozen=True)
class Undo:
    routes: tuple
    owner: object


class State:
    def __init__(self, instance: Instance, routes):
        self.p = instance
        audit(instance, routes)
        self._routes = [list(r) for r in routes if r]
        self._routes += [[] for _ in range(instance.max_vehicles - len(self._routes))]
        self.costs, self.loads, self.prefix = {}, {}, {}
        self.positions = {}
        self.owner, self.revision, self._undo = object(), 0, []
        for r in range(len(self._routes)):
            self._refresh(r)
        self.total = sum(self.costs.values())

    @property
    def routes(self):
        return tuple(tuple(r) for r in self._routes)

    def _refresh(self, r):
        route = self._routes[r]
        self.costs[r] = route_cost(self.p, route)
        prefix = [0]
        for i, v in enumerate(route):
            prefix.append(prefix[-1] + int(self.p.demands[v]))
            self.positions[v] = (r, i)
        self.loads[r], self.prefix[r] = prefix[-1], prefix

    def check(self):
        if audit(self.p, self._routes) != self.total:
            raise AssertionError("objective cache mismatch")
        for r, route in enumerate(self._routes):
            if self.costs[r] != route_cost(self.p, route):
                raise AssertionError("route cost cache mismatch")
            prefix = [0]
            for i, v in enumerate(route):
                prefix.append(prefix[-1] + int(self.p.demands[v]))
                if self.positions.get(v) != (r, i):
                    raise AssertionError("position cache mismatch")
            if prefix != self.prefix[r] or prefix[-1] != self.loads[r]:
                raise AssertionError("load cache mismatch")

    def _valid(self, m):
        if not (0 <= m.r < len(self._routes) and 0 <= m.s < len(self._routes)):
            raise ValueError("invalid route index")
        a, b = self._routes[m.r], self._routes[m.s]
        if m.kind == "two_opt":
            ok = m.r == m.s and 0 <= m.i < m.j - 1 and m.j <= len(a)
        elif m.kind == "relocate":
            ok = m.r != m.s and 0 <= m.i < len(a) and 0 <= m.j <= len(b)
        elif m.kind == "swap":
            ok = m.r != m.s and 0 <= m.i < len(a) and 0 <= m.j < len(b)
        elif m.kind == "tail":
            ok = m.r != m.s and 0 <= m.i <= len(a) and 0 <= m.j <= len(b)
        else:
            ok = False
        if not ok:
            raise ValueError("invalid move or unsupported same-route operator")

    def changed_routes(self, m):
        self._valid(m)
        a, b = self._routes[m.r], self._routes[m.s]
        if m.kind == "two_opt":
            return {m.r: a[:m.i] + a[m.i:m.j][::-1] + a[m.j:]}
        if m.kind == "relocate":
            return {m.r: a[:m.i] + a[m.i+1:], m.s: b[:m.j] + [a[m.i]] + b[m.j:]}
        if m.kind == "swap":
            aa, bb = a.copy(), b.copy()
            aa[m.i], bb[m.j] = bb[m.j], aa[m.i]
            return {m.r: aa, m.s: bb}
        return {m.r: a[:m.i] + b[m.j:], m.s: b[:m.j] + a[m.i:]}

    def evaluate(self, m: Move, mode="incremental"):
        if mode not in {"full", "incremental"}:
            raise ValueError("unknown evaluator")
        self._valid(m)
        p, a, b = self.p, self._routes[m.r], self._routes[m.s]
        d, q = p.costs, p.demands
        if mode == "full":
            changed = self.changed_routes(m)
            if any(sum(int(q[v]) for v in r) > p.capacity for r in changed.values()):
                return None
            delta = sum(route_cost(p, r) - self.costs[k] for k, r in changed.items())
            return Evaluation(m, int(delta), self.revision, self.owner)
        before = lambda route, i: route[i-1] if i else 0
        after = lambda route, i: route[i] if i < len(route) else 0
        if m.kind == "two_opt":
            aa, x, y, bb = before(a, m.i), a[m.i], a[m.j-1], after(a, m.j)
            delta = d[aa, y] + d[x, bb] - d[aa, x] - d[y, bb]
        elif m.kind == "relocate":
            x = a[m.i]
            if self.loads[m.s] + q[x] > p.capacity:
                return None
            aa, ab = before(a, m.i), after(a, m.i+1)
            ba, bb = before(b, m.j), after(b, m.j)
            delta = d[aa, ab]-d[aa, x]-d[x, ab]+d[ba, x]+d[x, bb]-d[ba, bb]
        elif m.kind == "swap":
            x, y = a[m.i], b[m.j]
            if max(self.loads[m.r]-q[x]+q[y], self.loads[m.s]-q[y]+q[x]) > p.capacity:
                return None
            aa, ab = before(a, m.i), after(a, m.i+1)
            ba, bb = before(b, m.j), after(b, m.j+1)
            delta = (d[aa, y]+d[y, ab]-d[aa, x]-d[x, ab]
                     + d[ba, x]+d[x, bb]-d[ba, y]-d[y, bb])
        else:
            pa, pb = self.prefix[m.r][m.i], self.prefix[m.s][m.j]
            if max(pa+self.loads[m.s]-pb, pb+self.loads[m.r]-pa) > p.capacity:
                return None
            aa, ab = before(a, m.i), after(a, m.i)
            ba, bb = before(b, m.j), after(b, m.j)
            delta = d[aa, bb]+d[ba, ab]-d[aa, ab]-d[ba, bb]
        return Evaluation(m, int(delta), self.revision, self.owner)

    def apply(self, ev: Evaluation):
        if ev.owner is not self.owner or ev.revision != self.revision:
            raise ValueError("foreign or stale evaluation")
        changed = self.changed_routes(ev.move)
        actual = sum(route_cost(self.p, r)-self.costs[k] for k, r in changed.items())
        if actual != ev.delta or any(sum(int(self.p.demands[v]) for v in r) > self.p.capacity
                                     for r in changed.values()):
            raise ValueError("invalid evaluated delta")
        token = Undo(tuple((k, tuple(self._routes[k])) for k in changed), self.owner)
        for k, route in changed.items():
            self._routes[k] = route
            self._refresh(k)
        self.total += actual
        self.revision += 1
        self._undo.append(token)
        return token

    def undo(self, token):
        if token.owner is not self.owner or not self._undo or self._undo[-1] is not token:
            raise ValueError("undo must be local and LIFO")
        for k, route in token.routes:
            self._routes[k] = list(route)
            self._refresh(k)
        self.total = sum(self.costs.values())
        self._undo.pop()
        self.revision += 1

    def commit(self):
        self._undo.clear()


def propose(state: State, rng: random.Random, expanded=True):
    nonempty = [i for i, r in enumerate(state._routes) if r]
    r = rng.choice(nonempty)
    a = state._routes[r]
    kind = rng.choice(("two_opt", "relocate", "swap", "tail") if expanded else ("two_opt",))
    if kind == "two_opt":
        if len(a) < 2:
            return None
        i = rng.randrange(len(a)-1)
        return Move(kind, r, i, r, rng.randrange(i+2, len(a)+1))
    other = [s for s in range(len(state._routes)) if s != r]
    if not other:
        return None
    s = rng.choice(other)
    b = state._routes[s]
    if kind == "swap" and not b:
        return None
    i = rng.randrange(len(a)+1) if kind == "tail" else rng.randrange(len(a))
    j = rng.randrange(len(b)) if kind == "swap" else rng.randrange(len(b)+1)
    return Move(kind, r, i, s, j)
