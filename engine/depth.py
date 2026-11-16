"""Draft depth controller.

For each round, chooses how many draft tokens to try: the depth with the highest expected
(accepted tokens / round time). The Python counterpart of Ollama's
mlxrunner/speculate_depth.go logic:

- cost: round time per depth (ms), exponential average; only rounds at the same depth as the
  previous round enter the measurement (a depth change distorts the time).
- acceptance: conditional acceptance rate per position (given the earlier ones held), exponential average.
- selection goes at most up to "last trusted position + 1"; now and then a deep probe.

The object lives across requests: a new request starts from the proven depth.
"""

from __future__ import annotations

from bisect import insort
from typing import Dict, List

PROBE_INTERVAL = 4
PROBE_INTERVAL_MAX = 512
COST_ALPHA = 0.3
COST_CONSTRAINT = 0.25
ACCEPT_ALPHA = 0.1
ACCEPT_MIN_SAMPLES = 10


class Depth:
    def __init__(self, deepest: int = 0):
        self.cost: Dict[int, float] = {}
        self.depths: List[int] = []
        self.ratio: List[float] = [0.0]
        self.seen: List[int] = [0]
        self.deepest = deepest          # 0 = unlimited
        self.planned = 0                   # depth of the next round
        self.was_probe = False
        self.interval = PROBE_INTERVAL
        self.elapsed = 0
        self.last_pick = 0
        self.prev_d = -1
        self.probe_repeat = -1           # the probe depth is kept for one more round

    # -- public
    def pick(self) -> int:
        return self.planned

    def observe(self, d: int, accept: int, ms: float):
        if d == self.prev_d:
            self._observe_cost(d, ms)
        self.prev_d = d
        if d > 0:
            self._observe_accept(d, accept)
        self.planned = self._next_depth()

    def digest(self) -> str:
        m = " ".join(f"{d}:{self.cost[d]:.0f}ms" for d in self.depths)
        k = " ".join(f"{i}:{self._accept(i):.2f}" for i in range(1, self._limit_location() + 1))
        return f"cost [{m}] acceptance [{k}] choice {self._chosen()}"

    # -- cost model
    def _observe_cost(self, d, ms):
        if ms <= 0:
            return
        v = self.cost.get(d)
        if v is None:
            self.cost[d] = ms
            insort(self.depths, d)
        else:
            limit = COST_CONSTRAINT * v
            self.cost[d] = v + COST_ALPHA * max(-limit, min(limit, ms - v))

    def _cost(self, d) -> float:
        ds = self.depths
        if not ds:
            return 0.0
        if d <= ds[0]:
            return self.cost[ds[0]]
        if d >= ds[-1]:
            return self.cost[ds[-1]]
        for i in range(1, len(ds)):
            if d <= ds[i]:
                a, b = ds[i - 1], ds[i]
                t = (d - a) / (b - a)
                return self.cost[a] + t * (self.cost[b] - self.cost[a])
        return self.cost[ds[-1]]

    # -- acceptance model
    def _observe_accept(self, d, accept):
        for i in range(1, d + 1):
            if accept < i - 1:
                break
            while len(self.seen) <= i:
                self.ratio.append(0.0)
                self.seen.append(0)
            result = 1.0 if accept >= i else 0.0
            if self.seen[i] == 0:
                self.ratio[i] = result
            else:
                self.ratio[i] += ACCEPT_ALPHA * (result - self.ratio[i])
            self.seen[i] += 1

    def _accept(self, i) -> float:
        if 1 <= i < len(self.seen) and self.seen[i] >= ACCEPT_MIN_SAMPLES:
            return self.ratio[i]
        for j in range(i - 1, 0, -1):
            if j < len(self.seen) and self.seen[j] >= ACCEPT_MIN_SAMPLES:
                return self.ratio[j]
        return 1.0

    def _expected(self, n) -> float:
        total, mul_pow2 = 1.0, 1.0
        for k in range(1, n + 1):
            mul_pow2 *= self._accept(k)
            total += mul_pow2
        return total

    def _limit_location(self) -> int:
        f = 0
        for i in range(1, len(self.seen)):
            if self.seen[i] >= ACCEPT_MIN_SAMPLES:
                f = i
            else:
                break
        return f

    def _limit(self) -> int:
        s = self._limit_location() + 1
        return min(s, self.deepest) if self.deepest > 0 else s

    # -- selection
    def _chosen(self) -> int:
        if len(self.cost) < 2:
            return 0
        best, best_ev = 0, self._expected(0) / self._cost(0)
        for n in range(1, self._limit() + 1):
            ev = self._expected(n) / self._cost(n)
            if ev > best_ev:
                best, best_ev = n, ev
        return best

    def _next_depth(self) -> int:
        # The probe runs two rounds in a row: since cost is measured only in the second
        # round at the same depth, a single-round probe would never refresh that depth's
        # time; an expensive value measured once while the machine was loaded would stay forever.
        if self.probe_repeat >= 0:
            d, self.probe_repeat = self.probe_repeat, -1
            return d
        pick = self._chosen()
        if pick != self.last_pick:
            self.interval = PROBE_INTERVAL
            self.elapsed = 0
            self.last_pick = pick
        elif self.was_probe:
            self.interval = min(self.interval * 2, PROBE_INTERVAL_MAX)
        self.was_probe = False

        self.elapsed += 1
        if self.elapsed >= self.interval:
            self.elapsed = 0
            probe = min(pick + 1, self._limit())
            if probe != pick:
                self.was_probe = True
                self.probe_repeat = probe
                return probe
        for n in range(0, self._limit() + 1):   # the shallowest unmeasured depth
            if n not in self.cost:
                return n
        return pick
