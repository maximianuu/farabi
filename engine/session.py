"""Chat session: a conversation's state inside the engine and its cached continuation.

A session holds: the main model's cache (keys/values in the 16 attention layers,
recurrent state in the 48 linear layers), the MTP head's cache, the cached
token sequence, and the hidden state of the last slot.

The state of the linear layers cannot be rewound; so "snapshots" are taken
wherever a return may be needed (the state of 48 layers ≈ 151 MB; the attention cache is not copied,
only its length is pulled back). Snapshots are taken in two places: the end of the system block
(new chats fork from here with the same system prompt) and the start of the last generation
(the same chat's next message continues from here).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

from .core import LOCATION, Engine, _gdn_rollback
from .sampling import Sampler


@dataclass
class Snapshot:
    location: int            # number of tokens in the cache
    gdn: list             # [(convolution state, delta state)] × 48
    hidden: mx.array       # hidden state of the position-1 slot
    shift: int
    n_images: int


@dataclass
class Stats:
    prompt_tok: int = 0
    read_tok: int = 0
    prompt_secs: float = 0.0
    generated: int = 0
    generate_secs: float = 0.0
    kind: int = 0
    draft: int = 0
    accept: int = 0
    finish: str = "stop"


class Session:
    def __init__(self, m: Engine):
        self.ct, self.cm = m.session_cache()
        self.gdn = [c for c in self.ct if not isinstance(c, KVCache)]
        self.kv = [c for c in self.ct if isinstance(c, KVCache)]
        self.tok: List[int] = []
        self.hidden: Optional[mx.array] = None
        self.shift = 0
        self.image_list: List[str] = []      # digests of the images in the cache, in order
        self.snapshots: List[Snapshot] = []
        self.latest = time.time()
        self.transient = False                # one-shot request (schema-constrained / tool connector): kept in a small separate pool
        self.root_limit = 0                 # snapshots up to this position (intermediate checkpoints in the system block) are not pruned
        self.busy = False                # a job is reading/generating in this session (possibly paused): the pool does not reuse it in place and does not discard it (2026-11-12)

    def states(self):
        d = []
        for c in self.ct + (self.cm or []):
            if isinstance(c, KVCache):
                if c.keys is not None:
                    d += [c.keys, c.values]
            else:
                d += [x for x in c.cache if x is not None]
        return d

    def take_snapshot(self):
        self.snapshots = [a for a in self.snapshots if a.location < len(self.tok)]
        # 2026-11-11: intermediate checkpoints in the system block (<= root_limit) are always kept — requests where the END of the system prompt changes (tool list, recipe
        # note) fork from the nearest checkpoint instead of reading from the start. Of the later ones, the root (first) + the newest two are kept; each is ≈ 151 MB, forks share them.
        root = [a for a in self.snapshots if a.location <= self.root_limit]
        beyond = [a for a in self.snapshots if a.location > self.root_limit]
        if len(beyond) > 3:
            beyond = [beyond[0]] + beyond[-2:]
        self.snapshots = root + beyond
        self.snapshots.append(
            Snapshot(len(self.tok), [(c[0], c[1]) for c in self.gdn], self.hidden, self.shift, len(self.image_list))
        )

    def revert(self, a: Snapshot):
        """Rolls the session back to where the snapshot was taken (in place)."""
        for c in self.kv:
            c.trim(c.offset - a.location)
        if self.cm:
            self.cm[0].trim(self.cm[0].offset - (a.location - 1))
        for c, (ev, dl) in zip(self.gdn, a.gdn):
            c[0], c[1] = ev, dl
        self.tok = self.tok[: a.location]
        self.hidden, self.shift = a.hidden, a.shift
        self.image_list = self.image_list[: a.n_images]
        self.snapshots = [x for x in self.snapshots if x.location <= a.location]

    def fork(self, m: Engine, a: Snapshot) -> "Session":
        """A new, independent session from the snapshot (the source is left untouched)."""
        y = Session(m)
        for tgt, source in zip(y.kv, self.kv):
            tgt.keys = source.keys[..., : a.location, :]
            tgt.values = source.values[..., : a.location, :]
            tgt.offset = a.location
        if self.cm:
            y.cm[0].keys = self.cm[0].keys[..., : a.location - 1, :]
            y.cm[0].values = self.cm[0].values[..., : a.location - 1, :]
            y.cm[0].offset = a.location - 1
        for c, (ev, dl) in zip(y.gdn, a.gdn):
            c[0], c[1] = ev, dl
        y.tok = self.tok[: a.location]
        y.hidden, y.shift = a.hidden, a.shift
        y.image_list = self.image_list[: a.n_images]
        y.snapshots = [Snapshot(x.location, x.gdn, x.hidden, x.shift, x.n_images) for x in self.snapshots if x.location <= a.location]      # earlier checkpoints pass too (arrays are shared)
        if not any(x.location == a.location for x in y.snapshots):
            y.snapshots.append(Snapshot(a.location, a.gdn, a.hidden, a.shift, a.n_images))
        y.root_limit = min(self.root_limit, a.location)
        mx.eval(y.states())
        return y

    def memory(self) -> int:
        """Approximate bytes: attention caches + snapshots."""
        b = 0
        for c in self.kv + (self.cm or []):
            if c.keys is not None:
                ratio = c.offset / max(c.keys.shape[2], 1)
                b += int((c.keys.nbytes + c.values.nbytes) * ratio)
        for a in self.snapshots:
            b += sum(ev.nbytes + dl.nbytes for ev, dl in a.gdn)
        return b


class Cancelled(Exception):
    pass


def step_forward(m: Engine, o: Session, ids: List[int], image=None, snapshot_places=(), chunk: int = 2048,
           halt: Optional[Callable[[], bool]] = None):
    """Reads the part of ids that is not in the session (from len(o.tok) to the end).

    image = (embeddings (1,n,H), position (3,1,n), shift) — if the part to be read contains an image.
    snapshot_places: a snapshot is taken on reaching these positions.
    """
    begin, n = len(o.tok), len(ids)
    if begin >= n:
        return
    p = mx.array(ids)[None]
    embedding = location = None
    if image is not None:
        embedding, location, shift = image
    limits = sorted({x for x in snapshot_places if begin < x < n} | {n})
    s = begin
    try:
        for limit in limits:
            while s < limit:
                e = min(s + chunk, limit)
                if embedding is None:
                    LOCATION.arr, LOCATION.shift = None, o.shift
                    g = m._hidden(p[:, s:e], o.ct)
                else:
                    LOCATION.arr = location[:, :, s:e]
                    g = m.body(p[:, s:e], cache=o.ct, input_embeddings=embedding[:, s:e, :])
                outputs = [g[:, -1:, :]]
                if o.cm is not None:
                    # head pairs: slot S = (token S+1, the main model's hidden state at S)
                    if s > 0:
                        t0, y0 = s, s - 1
                        hid = o.hidden if e - s == 1 else mx.concatenate([o.hidden, g[:, : e - s - 1, :]], axis=1)
                    else:
                        t0, y0 = 1, 0
                        hid = g[:, : e - 1, :]
                    amount = e - t0
                    if amount > 0:
                        if embedding is None:
                            eg = m._embed(p[:, t0:e])
                        else:
                            eg = embedding[:, t0:e, :]
                            LOCATION.arr = location[:, :, y0 : y0 + amount]
                        outputs.append(m.mtp(eg, hid, o.cm[0])[:, -1:, :])
                mx.eval(outputs, o.states())
                o.hidden = outputs[0]
                o.tok.extend(ids[s:e])
                s = e
                mx.clear_cache()
                if halt is not None and halt():
                    raise Cancelled()
            if limit < n:
                o.take_snapshot()
    finally:
        LOCATION.arr = None
    if image is not None:
        o.shift = shift


def generate(m: Engine, o: Session, max_n: int, sampler: Sampler, stats: Stats,
         halt: Optional[Callable[[], bool]] = None, constraint=None, no_draft: bool = False,
         trace: Optional[list] = None, force: Optional[Callable[[List[int]], Optional[int]]] = None) -> Iterator[List[int]]:
    """Generates from the end of the session; yields a list of new tokens per round.

    When done, the session is left consistent: the cache holds the prompt + everything generated EXCEPT the
    LAST token (the last token — usually the end marker — is read as part of the next prompt). constraint: schema enforcer (engine/constraint.py); works together with drafts.
    """
    ct, cm, gdn, kv = o.ct, o.cm, o.gdn, o.kv
    LOCATION.arr, LOCATION.shift = None, o.shift
    controller = m.depth

    def pick(lg):
        if sampler.is_greedy:
            return mx.argmax(lg, axis=-1), None, None
        cand, p = sampler.dist(lg)
        return sampler.draw(cand, p), cand, p

    t_begin = time.perf_counter()
    lg = m._logit(o.hidden)[:, -1, :]
    if constraint is not None:
        lg = constraint.apply(lg)
    current, _, _ = pick(lg)
    mx.eval(current)
    si = current.item()
    if constraint is not None:
        constraint.consume([si], m.eos)
    generated = 1
    pending_tok, pending_hid = current[None], o.hidden
    finished = si in m.eos
    all_tok = [si]                      # all generated tokens (for loop protection)
    yield [si]

    try:
        while not finished and generated < max_n:
            if halt is not None and halt():
                stats.finish = "cancelled"
                break
            remaining = max_n - generated
            d = 0 if no_draft else max(0, min(controller.pick(), remaining - 1))     # no_draft: diagnostics (plain decoding)
            # loop protection (supplied by the server): if a token comes around, the next token is not sampled but forced to be it (e.g. </think> or the end token)
            forced = force(all_tok) if force is not None else None
            if forced is not None:
                d = 0
            t0 = time.perf_counter()

            if d == 0:
                g = m._hidden(current[None], ct)
                lg = m._logit(g)[:, -1, :]
                if constraint is not None:
                    lg = constraint.apply(lg)
                y, _, _ = pick(lg)
                if forced is not None:
                    y = mx.array([forced])
                mx.eval(y, g, o.states())
                o.tok.append(si)
                if pending_tok.shape[1] >= 64:        # in a long plain run, write the pending pairs to the head
                    h = m.mtp(m._embed(pending_tok), pending_hid, cm[0])
                    mx.eval(h, o.states())
                    pending_tok, pending_hid = y[None], g
                else:
                    pending_tok = mx.concatenate([pending_tok, y[None]], axis=1)
                    pending_hid = mx.concatenate([pending_hid, g], axis=1)
                current, si = y, y.item()
                if constraint is not None:
                    constraint.consume([si], m.eos)
                new, accept = [si], 0
            else:
                # 1) write the pending pairs in a single pass; the last row yields the first draft
                h = m.mtp(m._embed(pending_tok), pending_hid, cm[0])[:, -1:, :]
                draft, q_cand, q_prob = [], [], []
                for i in range(d):
                    if i > 0:
                        h = m.mtp(m._embed(draft[-1][None]), h, cm[0])
                    y, cand, p = pick(m._logit(h)[:, -1, :])
                    draft.append(y)
                    if cand is not None:
                        q_cand.append(cand)
                        q_prob.append(p)
                draft_arr = mx.stack(draft, axis=1)

                # 2) main model: [current, drafts] in a single pass
                for c in gdn:
                    c.record_on = True
                g = m._hidden(mx.concatenate([current[None], draft_arr], axis=1), ct)
                lg = m._logit(g)[0]
                for c in gdn:
                    c.record_on = False
                if constraint is not None:
                    # schema: once the drafts are known, each verification row is masked with its own schema state
                    mx.eval(draft_arr, lg, g, o.states())
                    lg = constraint.apply_rows(lg, draft_arr[0].tolist())
                if sampler.is_greedy:
                    tgt = mx.argmax(lg, axis=-1)
                    mx.eval(draft_arr, tgt, g, o.states())
                    tl, hl = draft_arr[0].tolist(), tgt.tolist()
                    accept = 0
                    while accept < d and tl[accept] == hl[accept]:
                        accept += 1
                    following = hl[accept]
                else:
                    p_cand, p_prob = sampler.dist(lg)
                    qk, qp = mx.concatenate(q_cand, axis=0), mx.concatenate(q_prob, axis=0)
                    mx.eval(draft_arr, qk, qp, p_cand, p_prob, g, o.states())
                    tl = draft_arr[0].tolist()
                    accept, following = sampler.accept_reject(tl, np.array(qk), np.array(qp), np.array(p_cand), np.array(p_prob))
                # if the accepted drafts contain the end marker, stop there: the end
                # marker does not enter the cache (it is read together with the next prompt)
                observed = accept
                for i in range(accept):
                    if tl[i] in m.eos:
                        accept, following = i, tl[i]
                        break
                new = tl[:accept] + [following]
                if constraint is not None:
                    constraint.consume(new, m.eos)
                if trace is not None:      # diagnostics: per round (drafts, accepted count, the one that follows)
                    trace.append((tl, accept, following))

                # 3) roll back: the main model cache keeps only [current + accepted]
                if accept < d:
                    for c in kv:
                        c.trim(d - accept)
                    for c in gdn:
                        _gdn_rollback(c, accept + 1)
                for c in gdn:
                    c.record = None
                if d > 1:
                    cm[0].trim(d - 1)
                o.tok.append(si)
                o.tok.extend(tl[:accept])
                pending_tok = mx.array([new])
                pending_hid = g[:, : accept + 1, :]
                current, si = mx.array([following]), following
                stats.draft += d
                stats.accept += observed
                accept = observed

            stats.kind += 1
            controller.observe(d, accept, 1000 * (time.perf_counter() - t0))
            for i, b in enumerate(new):
                if b in m.eos:
                    new = new[: i + 1]
                    finished = True
                    break
            generated += len(new)
            all_tok.extend(new)
            # 0.4.4 (2026-11-14, measured): context arrays are reallocated every 256 tokens (KVCache.step); the old arrays fall into MLX's buffer
            # memory and were not released until the request finished -> with a 21k-token context, generating 3k tokens pushed the process to 51 GB.
            # Same rule as in Gemma (gemma.generate): release every 64 rounds.
            if stats.kind % 64 == 0:
                mx.clear_cache()
            yield new
        if not finished and stats.finish == "stop":
            stats.finish = "length"
    finally:
        # settle: write the pending pairs to the head except the last pair -> the session stays in "end of prompt" form
        k = pending_tok.shape[1]
        if k > 1:
            LOCATION.arr, LOCATION.shift = None, o.shift
            h = m.mtp(m._embed(pending_tok[:, : k - 1]), pending_hid[:, : k - 1, :], cm[0])
            mx.eval(h, o.states())
        o.hidden = pending_hid[:, -1:, :]
        mx.eval(o.hidden)
        for c in gdn:
            c.record_on, c.record = False, None
        stats.generated = generated
        stats.generate_secs = time.perf_counter() - t_begin
        o.latest = time.time()


class LoopGuard:
    """Repetition loop protection (2026-11-11; measured: on one travel question the model got into a loop while thinking and generated 30,816 tokens, 241 s).

    If the tail of the generated tokens is exactly periodic (the same chunk repeats back to back), it counts as a loop:
      while thinking  -> `</think>` is forced once (the model moves on to the answer with what it has); the same happens if thinking runs too long (limit; default 3,000 tokens ≈ 30 s, changeable per request with options.thinking_limit);
      in the answer or the second time -> the end token is forced.
    is_thinking: are we currently inside the thinking block (the state of the parser in the server)."""

    def __init__(self, think_close: Optional[int], finish: int, is_thinking: Callable[[], bool], limit: int = 3000):
        self.think_close, self.finish, self.is_thinking, self.limit = think_close, finish, is_thinking, limit
        self.latest = 0
        self.closed = False
        self.event: List[str] = []

    @staticmethod
    def period(ids: List[int], span: int, reps: int, max_n: int = 256) -> Optional[int]:
        a = np.array(ids[-max(span, reps * max_n):])
        for p in range(1, max_n + 1):
            L = max(reps * p, span)
            if L > len(a):
                break
            seg = a[-L:]
            if np.array_equal(seg[:-p], seg[p:]):
                return p
        return None

    def __call__(self, ids: List[int]) -> Optional[int]:
        n = len(ids)
        if n - self.latest < 48 or n < 160:
            return None
        self.latest = n
        think = self.is_thinking() and not self.closed
        if think:
            p = self.period(ids, 160, 3)
            if p is None and n < self.limit:
                return None
            self.closed = True
            self.event.append(f"thinking disabled ({'loop p=%d' % p if p else 'length'} @{n})")
            return self.think_close if self.think_close is not None else self.finish
        p = self.period(ids, 480, 6)
        if p is None:
            return None
        self.event.append(f"stopped (loop p={p} @{n})")
        return self.finish


# ---------------------------------------------------------------- cache


def _common_prefix(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    x, y = np.asarray(a[:n]), np.asarray(b[:n])
    return int(np.argmax(x != y))


class SessionCache:
    """Holds the sessions; finds the one matching the longest prefix for a new prompt."""

    def __init__(self, m: Engine, max_sessions: int = 8, max_bytes: int = 12 << 30, min_len: int = 64):
        self.m = m
        self.sessions: List[Session] = []
        self.max_sessions = max_sessions
        self.max_bytes = max_bytes
        self.min_len = min_len
        self.max_transient = 6             # 2026-11-11: 2 -> 6 (the client app sends three one-shot requests every round: decision extraction, title, suggestion; the decision prompt is 2,450 tokens)

    def pick(self, ids: List[int], digests: List[str], image_places: List[int], transient: bool = False):
        """Returns: (session, how) — how: "resumed" | "rewound" | "forked" | "new".

        transient: a one-shot request. It does not modify a persistent chat's session in place; if there is a shared prefix
        it forks from it (the chat's own state is not disturbed).
        """
        best = (0, None, None, None)     # (position, session, snapshot or None=live, in place?)
        for o in self.sessions:
            l = _common_prefix(o.tok, ids)
            # images: different images of the same size produce the same tokens -> verify via digest
            for i, place in enumerate(image_places):
                if place >= l:
                    break
                if i >= len(o.image_list) or o.image_list[i] != digests[i]:
                    l = min(l, place)
                    break
            # busy: the session of a paused job (the preempting request is selecting). Its live state is a half-finished generation -> it is not resumed or rewound;
            # it can only be forked from its immutable snapshots (the fork does not touch the source).
            busy = getattr(o, "busy", False)
            if l == len(o.tok) and l > best[0] and not busy:
                best = (l, o, None, True)
            for a in o.snapshots:
                if a.location <= l and a.location > best[0]:
                    in_place = a is o.snapshots[-1] and not busy
                    best = (a.location, o, a, in_place)
        location, o, a, in_place = best
        if o is None or location < self.min_len:
            y = Session(self.m)
            self.sessions.append(y)
            return y, "new"
        no_touch = transient and not o.transient          # a persistent chat's session is not modified in place
        if a is None:
            if not no_touch:
                return o, "continued"
            o.take_snapshot()
            a, in_place = o.snapshots[-1], False
        if in_place and not no_touch:
            o.revert(a)
            return o, "rewind"
        y = o.fork(self.m, a)
        self.sessions.append(y)
        return y, "fork"

    def release(self, o: Session):
        if o in self.sessions:
            self.sessions.remove(o)

    def prune(self, protect: Optional[Session] = None):
        # sessions of one-shot requests: the newest two are kept (so that consecutive questions about the same screenshot do not re-read the image)
        transient = sorted((x for x in self.sessions if x.transient), key=lambda x: x.latest)
        for x in transient[:-self.max_transient]:
            if x is not protect and not getattr(x, "busy", False):
                self.sessions.remove(x)

        def carrying():
            persistent = sum(1 for x in self.sessions if not x.transient)
            return persistent > self.max_sessions or self.total_memory() > self.max_bytes
        while carrying():
            discardable = [x for x in self.sessions if x is not protect and not getattr(x, "busy", False)]      # the session of a paused job is not discarded
            candidates = [x for x in discardable if not x.transient] or discardable
            if not candidates:
                break
            self.sessions.remove(min(candidates, key=lambda x: x.latest))
        mx.clear_cache()

    def total_memory(self) -> int:
        """Approximate bytes of all sessions; snapshot arrays shared between forks are counted once."""
        b, seen = 0, set()
        for o in self.sessions:
            for c in o.kv + (o.cm or []):
                if c.keys is not None:
                    b += int((c.keys.nbytes + c.values.nbytes) * (c.offset / max(c.keys.shape[2], 1)))
            for a in o.snapshots:
                for ev, dl in a.gdn:
                    for d in (ev, dl):
                        if id(d) not in seen:
                            seen.add(id(d)); b += d.nbytes
        return b

    def digest(self) -> str:
        return " ".join(f"[{len(o.tok)}t {o.memory() >> 20}MB {len(o.snapshots)}a]" for o in self.sessions)
