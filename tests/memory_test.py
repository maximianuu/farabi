#!/usr/bin/env python3
"""0.4.4 memory fixes, tested without a model: root checkpoint placement, pool budgets (based on real usage), applying the budget to both pools
(each pool keeps its newest session), releasing memory when the job ends on the cancel / error path, disk upper limit.
Run: <venv python> tests/memory_test.py <engine root>      (mlx is needed only for the disk test; the other tests also run with the standard library)"""
import os, sys, types

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("FARABI_NETWORK_LOCK", "test")       # do not let the server relaunch itself in the sandbox when imported
import engine.server as S

elapsed = total = 0
def check(name, condition, detail=""):
    global elapsed, total
    total += 1; elapsed += bool(condition)
    print(("PASS " if condition else "FAIL ") + name + (("  " + str(detail)) if detail and not condition else ""))

# ── 1) root checkpoint placement ──
kd = S.root_checkpoints
old = lambda k: set(range(512, k, 512)) | {k}
check("k < 512: no checkpoints", kd(511) == set() and kd(0) == set())
check("k = 512: block end only", kd(512) == {512}, kd(512))
check("short prompt (1427): same as before (512, 1024, end)", kd(1427) == {512, 1024, 1427}, sorted(kd(1427)))
check("short prompt (2000): same as before", kd(2000) == {512, 1024, 1536, 2000}, sorted(kd(2000)))
check("in every block with k ≤ 4096 the set is the SAME as before (template prompts)", all(kd(k) == old(k) for k in list(range(512, 4097, 37)) + [3095, 3600, 4096]))
check("template prompt (3600): 512 and 1024 are kept (review finding)", {512, 1024, 1536} <= kd(3600), sorted(kd(3600)))
check("in a long block the 512-spaced checkpoints within the first 2048 are kept too", all({512, 1024, 1536, 2048} <= kd(k) for k in (4200, 9300, 16500, 100000)))
u = kd(9300)
check("client app system prompt (9300): 19 → 10 checkpoints", u == {512, 1024, 1536, 2048, 4096, 6144, 8192, 8704, 9216, 9300} and len(old(9300)) == 19, sorted(u))
check("every multiple of 2048 is kept (disk._chosen writes these)", all(x in kd(15000) for x in range(2048, 15000, 2048)))
for k in (1427, 3095, 9300, 12361, 16500, 40000):
    s = sorted(kd(k))
    # the two checkpoints before the block end are at most 2 × 512 from the block end (disk._chosen: roots[-3:-1])
    check("k=%d: the last two intermediate checkpoints are ≤ 1024 from the block end" % k, len(s) < 3 or (k - s[-3] <= 1024 and k - s[-2] <= 512), s[-3:])
d = kd(100000)
check("single huge message (100000): checkpoint count is bounded (≤ 19; previously 196)", len(d) <= 19 and 100000 in d and len(old(100000)) == 196, len(d))
check("in a huge message only tail checkpoints remain after 2 × ROOT_CAP", all(x <= 2 * S.ROOT_CAP or x >= 100000 - S.TAIL_DENSE for x in d), sorted(d))
check("in a 16-32k block the gap between consecutive checkpoints is ≤ 4096 (second review: previously gaps up to k−1536)", all(max(b - a for a, b in zip([0] + sorted(kd(k)), sorted(kd(k)))) <= 4096 for k in range(16384, 34000, 613)))
check("in a block up to 16k the gap between consecutive checkpoints is ≤ 2048", all(max(b - a for a, b in zip([0] + sorted(kd(k)), sorted(kd(k)))) <= 2048 for k in range(512, 16385, 331)))
check("all checkpoints (except the block end) are multiples of 512", all(x % 512 == 0 for k in (3095, 9300, 100000) for x in kd(k) if x != k))
check("the new set is a subset of the old one", all(kd(k) <= old(k) for k in (600, 1427, 3095, 9300, 20000, 100000)))
check("checkpoint count does not grow without bound with block length (≤ 19, for every k)", max(len(kd(k)) for k in range(512, 262144, 997)) <= 19)

# ── 2) pool budgets ──
hb = S.pool_budgets
near = lambda a, b: abs(a - b) < 1e-6
check("Qwen only, 54 GB: 12 GB (old value)", hb(["qwen"], 54.0, 0.0) == {"qwen": 12.0}, hb(["qwen"], 54.0, 0.0))
check("Gemma only, 54 GB: 10 GB (old value)", hb(["gemma"], 54.0, 0.0) == {"gemma": 10.0}, hb(["gemma"], 54.0, 0.0))
check("Qwen only: 12 GB whatever the usage", hb(["qwen"], 54.0, 0.0, {"qwen": 11.0}) == {"qwen": 12.0})
FREE = 54.0 - 17 - 21 - 8
MARGIN = {"qwen": FREE * 12 / 22, "gemma": FREE * 10 / 22}
saturated = hb(["qwen", "gemma"], 54.0, 0.0, {"qwen": MARGIN["qwen"], "gemma": MARGIN["gemma"]})
check("both pools use their share: Qwen ≈ 4.4 + Gemma ≈ 3.6 = 8 GB", near(saturated["qwen"], MARGIN["qwen"]) and near(saturated["gemma"], MARGIN["gemma"]) and near(sum(saturated.values()), FREE), saturated)
q_empty = hb(["qwen", "gemma"], 54.0, 0.0, {"qwen": 0.0, "gemma": 5.0})
check("when the Qwen pool is empty Gemma gets 8 GB (same as the 0.4.3 formula; review finding)", near(q_empty["gemma"], 8.0), q_empty)
check("… Qwen's guaranteed share (≈ 4.4) still holds", near(q_empty["qwen"], MARGIN["qwen"]), q_empty)
old_gemma = lambda qk, oll=0.0: max(3.0, min(10.0, 54.0 - 21 - (17 + qk) - oll - 8))       # 0.4.3 server.py:843-844
check("Gemma's budget is never smaller than in 0.4.3 at any Qwen usage", all(hb(["qwen", "gemma"], 54.0, 0.0, {"qwen": qk / 4, "gemma": 2.0})["gemma"] >= old_gemma(qk / 4) - 1e-9 for qk in range(0, 49)))
g_empty = hb(["qwen", "gemma"], 54.0, 0.0, {"qwen": 6.0, "gemma": 0.0})
check("when the Gemma pool is empty Qwen gets 8 GB (previously a fixed 12 → the limit was exceeded)", near(g_empty["qwen"], 8.0), g_empty)
ok = True
for qk in range(0, 49):
    for gk in range(0, 41):
        ku = {"qwen": qk / 4, "gemma": gk / 4}
        b = hb(["qwen", "gemma"], 54.0, 0.0, ku)
        for a, dg in (("qwen", "gemma"), ("gemma", "qwen")):
            ok &= b[a] >= MARGIN[a] - 1e-9 and b[a] <= max(MARGIN[a], FREE - ku[dg]) + 1e-9 and S.POOL_FLOOR[a] <= b[a] <= S.POOL_CAP[a]
        # balance: when both pools are pruned to their budgets, total usage does not exceed the free space
        latest = {a: min(ku[a], b[a]) for a in b}
        b2 = hb(["qwen", "gemma"], 54.0, 0.0, latest)
        latest2 = {a: min(latest[a], b2[a]) for a in b2}
        ok &= sum(latest2.values()) <= FREE + 1e-9
check("in every usage combination: the share is guaranteed, budget ≤ free space − the other's usage, total after pruning ≤ 8 GB", ok)
tight = hb(["qwen", "gemma"], 54.0, 20.0, {"qwen": 1.0, "gemma": 1.0})
check("with Ollama holding 20 GB: both at the floor (3 GB)", tight == {"qwen": 3.0, "gemma": 3.0}, tight)
check("on a 32 GB machine with only Qwen (limit 24): the floor", hb(["qwen"], 24.0, 0.0) == {"qwen": 3.0}, hb(["qwen"], 24.0, 0.0))
check("empty list: empty dict", hb([], 54.0, 0.0) == {})

# ── 3) applying the budget to the pools (fake worker) ──
class O:
    def __init__(self, latest, name=""): self.latest, self.name = latest, name
class FakePool:
    def __init__(self, gb=0.0, sessions=()): self.max_bytes, self.pruned, self.gb, self.sessions = 12 << 30, [], gb, list(sessions)
    def prune(self, protect=None): self.pruned.append(protect)
    def total_memory(self): return int(self.gb * (1 << 30))
S.gpu_limit_gb = lambda: 54.0
S.ollama_loaded_gb = lambda: 0.0
S.log_fn = lambda *a, **k: None
GB = lambda scache: scache.max_bytes / (1 << 30)
q1, q2 = O(10, "q-old"), O(20, "q-new")
worker = types.SimpleNamespace(engine=object(), gemma=None, session_cache=FakePool(6.0, [q1, q2]), gemma_session_cache=None, _ollama_at=(0.0, 0.0))
S.Worker._apply_budget(worker, protect=q1)
check("only Qwen loaded: budget 12 GB, prune was called protecting the request's session", worker.session_cache.max_bytes == 12 << 30 and worker.session_cache.pruned == [q1])
# BEFORE Gemma loads: Qwen's pool shrinks (Gemma's pool does not exist yet)
S.Worker._apply_budget(worker, to_load="gemma")
check("Gemma about to load: Qwen pool 12 → 8 GB before the weights are allocated", near(GB(worker.session_cache), 8.0) and worker.session_cache.pruned[-1] is q2, GB(worker.session_cache))
g1, g2 = O(5, "g-old"), O(30, "g-new")
worker.gemma, worker.gemma_session_cache = object(), FakePool(0.0, [])
S.Worker._apply_budget(worker)
check("Gemma loaded, its pool empty: Qwen 8 GB, Gemma at its guaranteed share (≈ 3.6)", near(GB(worker.session_cache), 8.0) and abs(GB(worker.gemma_session_cache) - MARGIN["gemma"]) < 0.01, (GB(worker.session_cache), GB(worker.gemma_session_cache)))
# Gemma request finished: the Gemma pool uses 3 GB → Qwen drops to 5 GB; Qwen's NEWEST session is kept in the Qwen pool
worker.gemma_session_cache.gb, worker.gemma_session_cache.sessions = 3.0, [g1, g2]
S.Worker._apply_budget(worker, protect=g2)
check("when the Gemma request finishes the Qwen pool goes 8 → 5 GB", near(GB(worker.session_cache), 5.0), GB(worker.session_cache))
check("other family's request: the newest Qwen session is kept in the Qwen pool (review finding)", worker.session_cache.pruned[-1] is q2)
check("… the request's own session is kept in the Gemma pool", worker.gemma_session_cache.pruned[-1] is g2)
# Qwen request finished (with an old session): Qwen keeps its own session, Gemma keeps its newest
S.Worker._apply_budget(worker, protect=q1)
check("Qwen request: the request's session is kept in the Qwen pool, Gemma's newest in the Gemma pool", worker.session_cache.pruned[-1] is q1 and worker.gemma_session_cache.pruned[-1] is g2)
# cancel: the session is no longer in the pool → that pool's newest is kept
S.Worker._apply_budget(worker, protect=O(99, "not-in-pool"))
check("if the protected session is not in the pool (cancelled request) each pool keeps its own newest", worker.session_cache.pruned[-1] is q2 and worker.gemma_session_cache.pruned[-1] is g2)
empty_pool = types.SimpleNamespace(engine=object(), gemma=None, session_cache=FakePool(0.0, []), gemma_session_cache=None, _ollama_at=(0.0, 0.0))
S.Worker._apply_budget(empty_pool)
check("protect = None on an empty pool (does not raise)", empty_pool.session_cache.pruned == [None])
worker.gemma, worker.gemma_session_cache = None, None
S.Worker._apply_budget(worker)
check("when Gemma is unloaded the Qwen pool is back to 12 GB", worker.session_cache.max_bytes == 12 << 30)
unpruned = types.SimpleNamespace(engine=object(), gemma=None, session_cache=types.SimpleNamespace(total_memory=lambda: 0), gemma_session_cache=None, _ollama_at=(0.0, 0.0))
try:
    S.Worker._apply_budget(unpruned); S.Worker._apply_budget(types.SimpleNamespace(engine=object(), gemma=None, session_cache=object(), gemma_session_cache=None, _ollama_at=(0.0, 0.0))); ok = True
except Exception as e:
    ok = False
check("a fake pool without prune / total_memory (queue_test) does not raise", ok)
empty = types.SimpleNamespace(engine=None, gemma=None, session_cache=None, gemma_session_cache=None, _ollama_at=(0.0, 0.0))
try:
    S.Worker._apply_budget(empty); S.Worker._release_memory(); ok = True
except Exception:
    ok = False
check("with no model loaded, applying the budget and releasing memory do not raise", ok)
# the Ollama poll is cached for 15 s
counter = [0]
def _oll(): counter[0] += 1; return 2.0
S.ollama_loaded_gb = _oll
single = types.SimpleNamespace(engine=object(), gemma=None, session_cache=FakePool(1.0, [q1]), gemma_session_cache=None, _ollama_at=(0.0, 0.0))
for _ in range(5): S.Worker._apply_budget(single)
check("Ollama is queried once per 15 s (1 poll in 5 calls)", counter[0] == 1 and near(GB(single.session_cache), 12.0), counter[0])
S.ollama_loaded_gb = lambda: 0.0

# budget before reading: Gemma has borrowed 8 GB, Qwen will read 50k tokens (≈ 3.4 GB) → Gemma's budget drops before the read
qh, gh0 = FakePool(0.5, [q1]), FakePool(8.0, [g1, g2])
ok_ = types.SimpleNamespace(engine=object(), gemma=object(), session_cache=qh, gemma_session_cache=gh0, _ollama_at=(0.0, 0.0))
ok_._apply_budget = lambda **kw: S.Worker._apply_budget(ok_, **kw)
S.Worker._apply_budget(ok_)
before_g = GB(gh0)
S.Worker._budget_before_read(ok_, "qwen", q1, 50000)
expected = max(MARGIN["gemma"], FREE - (0.5 + 50000 * 72.0 / (1 << 20)))
check("before reading: Gemma budget %.1f → %.1f GB (Qwen will read 50k tokens)" % (before_g, expected), near(before_g, 7.5) and abs(GB(gh0) - expected) < 0.01 and gh0.pruned[-1] is g2 and qh.pruned[-1] is q1, (before_g, GB(gh0)))
n0 = len(gh0.pruned)
S.Worker._budget_before_read(ok_, "qwen", q1, 2000)
check("before reading: no call for a short read (2k tokens)", len(gh0.pruned) == n0)
single_family = types.SimpleNamespace(engine=object(), gemma=None, session_cache=FakePool(1.0, [q1]), gemma_session_cache=None, _ollama_at=(0.0, 0.0))
single_family._apply_budget = lambda **kw: (_ for _ in ()).throw(AssertionError("must not be called"))
S.Worker._budget_before_read(single_family, "qwen", q1, 90000)
check("before reading: no call when only one family is loaded", True)
count2 = [0]
def _oll2(): count2[0] += 1; return 0.0
S.ollama_loaded_gb = _oll2
ok_._ollama_at = (0.0, 0.0)
S.Worker._budget_before_read(ok_, "gemma", g2, 60000)
check("the call before reading does not query Ollama (does not delay the response)", count2[0] == 0, count2[0])
S.ollama_loaded_gb = lambda: 0.0

# ── 3b) pruning with a real pool: a request from the other family does not drop the warm session ──
try:
    from engine.session import SessionCache
    class HO:
        def __init__(self, latest, gb): self.latest, self._gb, self.transient, self.busy = latest, gb, False, False
    class Pool(SessionCache):
        def __init__(self): self.sessions, self.max_sessions, self.max_bytes, self.max_transient = [], 8, 12 << 30, 6
        def total_memory(self): return int(sum(x._gb for x in self.sessions) * (1 << 30))
    hv = Pool(); old_o, warm = HO(1, 2.0), HO(2, 5.5)
    hv.sessions = [old_o, warm]
    gh = FakePool(3.0, [g2])
    ik = types.SimpleNamespace(engine=object(), gemma=object(), session_cache=hv, gemma_session_cache=gh, _ollama_at=(0.0, 0.0))
    S.Worker._apply_budget(ik, protect=g2)          # Gemma request finished; Qwen budget is 5 GB, the warm session alone is 5.5 GB
    check("real pruning: a Gemma request does not drop Qwen's WARM session that exceeds the budget, it drops the older one", hv.sessions == [warm], [x.latest for x in hv.sessions])
except ImportError as e:
    print("SKIPPED real pool test (no mlx):", e)

# ── 3c) cancel / error path: memory is released AFTER the job ends ──
import inspect
source = inspect.getsource(S.Worker)
for name in ("_chat_qwen", "chat_gemma"):
    g = inspect.getsource(getattr(S.Worker, name))
    check("%s: the read + generation cancel / error branches (4 places) do not release memory in place, they set the flag" % name, "self._release_memory()" not in g and g.count("self._release_pending = True") == 4)
    check("%s: before reading, the other pool's borrowed share is reclaimed" % name, g.index("self._budget_before_read(") < g.index("step_forward(m, o, ids"))
call = []
fake = types.SimpleNamespace(_release_pending=False, current=None, busy=False, engine=None, gemma=None, embedding=None, gemma_unloaded_at=0, qwen_unloaded_at=0,
                              _release_memory=lambda: call.append("release"))
class SingleJob:
    def __init__(self, jobs): self.jobs = list(jobs)
    def get(self, timeout=None):
        if not self.jobs: raise SystemExit
        return self.jobs.pop(0)
def cancel_job():
    try:
        raise RuntimeError("error while reading")           # the job function catches the exception itself (like job_ on the server)
    except RuntimeError:
        fake._release_pending = True
        call.append("branch")
fake.job_queue = SingleJob([types.SimpleNamespace(f=cancel_job), types.SimpleNamespace(f=lambda: call.append("regular"))])
try:
    S.Worker.run(fake)
except SystemExit:
    pass
check("run(): memory is released after a flagged job, not after a normal job", call == ["branch", "release", "regular"] and fake._release_pending is False, call)

# ── 4) disk: huge prefixes are not written ──
try:
    import engine.disk as D
    class Fake: pass
    o = Fake(); o.root_limit = 100000
    a = Fake(); a.location, a.n_images, a.shift = 100000, 0, 0
    o.snapshots = [a]; o.kv = []; o.cm = None
    D._ENV = ""
    check("disk: prefixes longer than 32768 are not written", D.save(o, "test-missing") is None and D.MAX_PREFIX == 32768)
    check("disk.INTERVAL = SPARSE_CHECKPOINT", D.INTERVAL == S.SPARSE_CHECKPOINT)
except ImportError as e:
    print("SKIPPED disk test (no mlx):", e)

print("RESULT: %d / %d" % (elapsed, total))
sys.exit(0 if elapsed == total else 1)
