"""Session pool: the session of a paused job (busy) is not used in place and not dropped — WITHOUT loading a model (small fake arrays).

Usage: <venv python> tests/pool_test.py <engine root>
"""
import sys

ROOT = sys.argv[1]
sys.path.insert(0, ROOT)
import mlx.core as mx
from mlx_lm.models.cache import KVCache

from engine import gemma as G
from engine import session as Q

elapsed = remaining = 0


def check(name, condition, detail=""):
    global elapsed, remaining
    if condition:
        elapsed += 1
    else:
        remaining += 1
    print(("  ✓ " if condition else "  ✗ ") + name + (f"  [{detail}]" if detail and not condition else ""))


class FakeQwen:
    def session_cache(self):
        return [KVCache()], None


class FakeGemma:
    def session_cache(self):
        return [KVCache()]


def fill(o, n, kv):
    """Fills the session with a fake state of n tokens (key/value: position number)."""
    d = mx.arange(n, dtype=mx.float32).reshape(1, 1, n, 1) * mx.ones((1, 1, 1, 4))
    for c in kv(o):
        c.keys, c.values, c.offset = d, d + 0.5, n
    o.tok = list(range(n))
    o.hidden = mx.zeros((1, 1, 4))


def build(mod, m, pool_class, kv):
    scache = pool_class(m)
    p = mod.Session(m)
    fill(p, 80, kv)
    p.take_snapshot()                      # snapshot at position 80 (the Qwen pool's minimum threshold is 64)
    fill(p, 160, kv)                # job advanced to 160 (half-way through generation)
    scache.sessions.append(p)
    return scache, p


for name, mod, m, pool, kv in (("Qwen pool", Q, FakeQwen(), Q.SessionCache, lambda o: o.kv),
                              ("Gemma pool", G, FakeGemma(), G.GemmaSessionCache, lambda o: o.full_attn)):
    print(name)
    long = list(range(160)) + [900, 901, 902]          # fully shared with P
    short = list(range(100)) + [900, 901, 902]          # only the first 100 are shared (snapshot is at 80)

    scache, p = build(mod, m, pool, kv)
    o, how = scache.pick(long, [], [])
    check("idle session: an exact match continues in place (old behaviour)", o is p and how == "continued", how)
    scache, p = build(mod, m, pool, kv)
    o, how = scache.pick(short, [], [])
    check("idle session: a partial match rewinds in place (old behaviour)", o is p and how == "rewind" and len(p.tok) == 80, (how, len(p.tok)))

    scache, p = build(mod, m, pool, kv)
    p.busy = True
    o, how = scache.pick(long, [], [])
    check("busy session: not handed out in place even on an exact match, forked from the snapshot", o is not p and how == "fork" and len(o.tok) == 80, (how, len(o.tok)))
    check("busy session left as it was (160 tokens, cache 160)", len(p.tok) == 160 and kv(p)[0].offset == 160 and len(p.snapshots) == 1, (len(p.tok), kv(p)[0].offset))
    check("the fork's cache is a copy of the first 80 positions", kv(o)[0].offset == 80 and kv(o)[0].keys.shape[2] == 80 and float(kv(o)[0].keys[0, 0, 79, 0]) == 79.0)
    # the content of the source must not change while the fork advances
    k0 = kv(o)[0]
    if how == "fork":
        k0.keys = mx.concatenate([k0.keys, mx.full((1, 1, 5, 4), -1.0)], axis=2)
        mx.eval(k0.keys)
    check("when the fork advanced, positions 80-84 of the busy session stayed the same", [float(kv(p)[0].keys[0, 0, i, 0]) for i in range(80, 85)] == [80.0, 81.0, 82.0, 83.0, 84.0])

    scache, p = build(mod, m, pool, kv)
    p.busy = True
    o, how = scache.pick(short, [], [])
    check("busy session: a partial match does not rewind, it forks", o is not p and how == "fork" and len(p.tok) == 160, (how, len(p.tok)))

    scache, p = build(mod, m, pool, kv)
    p.busy = True
    o, how = scache.pick(list(range(160)) + [7], [], [], transient=True)
    check("busy session + transient request: NO snapshot is taken from the live state", len(p.snapshots) == 1 and how == "fork" and len(o.tok) == 80, (len(p.snapshots), how))

    # pruning: a busy session is not dropped when the budget is exceeded
    scache, p = build(mod, m, pool, kv)
    p.busy, p.latest = True, 1.0                       # oldest → normally the first to be dropped
    q, r = mod.Session(m), mod.Session(m)
    fill(q, 30, kv)
    fill(r, 30, kv)
    q.latest, r.latest = 2.0, 3.0
    scache.sessions += [q, r]
    scache.max_sessions = 1
    scache.prune(protect=r)
    check("pruning: the busy session stayed although it was the oldest, the idle one was dropped", p in scache.sessions and q not in scache.sessions and r in scache.sessions, [x is p or x is q or x is r for x in scache.sessions])
    p.busy = False
    scache.prune(protect=r)
    check("once the flag is cleared the same session can be pruned", p not in scache.sessions and r in scache.sessions)

    # pruning of transient sessions
    scache = pool(m)
    scache.max_transient = 1
    g1, g2, g3 = mod.Session(m), mod.Session(m), mod.Session(m)
    for i, g in enumerate((g1, g2, g3)):
        fill(g, 10, kv)
        g.transient, g.latest = True, float(i)
    g1.busy = True
    scache.sessions += [g1, g2, g3]
    scache.prune(protect=None)
    check("in the transient pool too, the busy one stays", g1 in scache.sessions and g2 not in scache.sessions and g3 in scache.sessions)

print(f"\n{elapsed} passed, {remaining} failed")
sys.exit(1 if remaining else 0)
