"""Sampling and speculative acceptance.

GPU side: logits -> (the K most probable candidates, their probabilities). Temperature, top_k, top_p,
min_p are applied here; it stays lazy and joins the round's single eval.

CPU side: given the draft distribution q and the main model distribution p, the draft token x
is accepted with probability min(1, p(x)/q(x)); on rejection it is sampled from the residual distribution
(max(p-q, 0) normalized). Thus the distribution of the emitted tokens is the SAME as
plain sampling (standard speculative sampling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np


@dataclass
class Sampler:
    temperature: float = 0.0     # 0 = greedy
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0
    rng_seed: Optional[int] = None
    max_candidates: int = 256    # number of candidates scanned when top_k = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.rng_seed)
        if self.rng_seed is not None:
            mx.random.seed(self.rng_seed)

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0

    # -- GPU: (R, V) logits -> (R, K) ids and probabilities, in descending order
    def dist(self, logit: mx.array) -> Tuple[mx.array, mx.array]:
        K = self.top_k if self.top_k > 0 else self.max_candidates
        K = min(K, logit.shape[-1])
        logit = logit.astype(mx.float32)
        cand = mx.argpartition(-logit, K - 1, axis=-1)[..., :K]
        value = mx.take_along_axis(logit, cand, axis=-1)
        order = mx.argsort(-value, axis=-1)
        cand = mx.take_along_axis(cand, order, axis=-1)
        value = mx.take_along_axis(value, order, axis=-1)
        p = mx.softmax(value / self.temperature, axis=-1)
        if self.min_p > 0:
            p = mx.where(p >= self.min_p * p[..., :1], p, 0)
        if 0 < self.top_p < 1:
            previous = mx.cumsum(p, axis=-1) - p
            p = mx.where(previous < self.top_p, p, 0)
        p = p / p.sum(axis=-1, keepdims=True)
        return cand, p

    # -- GPU: lazy sample from the distribution (for the draft chain)
    def draw(self, cand: mx.array, p: mx.array) -> mx.array:
        i = mx.random.categorical(mx.log(mx.maximum(p, 1e-30)))
        return mx.take_along_axis(cand, i[..., None], axis=-1)[..., 0]

    # -- CPU
    def draw_cpu(self, cand: np.ndarray, p: np.ndarray) -> int:
        return int(cand[self.rng.choice(len(p), p=p / p.sum())])

    def accept_reject(
        self,
        draft: List[int],
        q_cand: np.ndarray, q_prob: np.ndarray,      # (d, K)
        p_cand: np.ndarray, p_prob: np.ndarray,      # (d+1, K)
    ) -> Tuple[int, int]:
        """Returns: (number of accepted drafts, the token that follows)."""
        d = len(draft)
        for i, x in enumerate(draft):
            qx = float(q_prob[i][q_cand[i] == x].sum())
            px = float(p_prob[i][p_cand[i] == x].sum())
            if qx > 0 and self.rng.random() < min(1.0, px / qx):
                continue
            # on rejection: the residual distribution max(p - q, 0)
            q_dict = dict(zip(q_cand[i].tolist(), q_prob[i].tolist()))
            remaining = np.array(
                [max(pp - q_dict.get(k, 0.0), 0.0) for k, pp in zip(p_cand[i].tolist(), p_prob[i].tolist())]
            )
            if remaining.sum() <= 0:      # p <= q everywhere: only a numerical corner case
                return i, self.draw_cpu(p_cand[i], p_prob[i])
            return i, self.draw_cpu(p_cand[i], remaining)
        return d, self.draw_cpu(p_cand[d], p_prob[d])
