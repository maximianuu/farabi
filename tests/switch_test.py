#!/usr/bin/env python3
"""0.4.5 test without a model: the rule for unloading the other family when the model changes (Worker._release_other, Worker.chat, JobQueue.has_family).
Run: <venv python> tests/switch_test.py <engine root>"""
import os, sys, time, types

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("FARABI_NETWORK_LOCK", "test")
import engine.server as S

elapsed = total = 0
def check(name, condition, detail=""):
    global elapsed, total
    total += 1; elapsed += bool(condition)
    print(("PASS " if condition else "FAIL ") + name + (("  " + str(detail)) if detail and not condition else ""))

logs = []
S.log_fn = lambda *a, **k: logs.append(" ".join(str(x) for x in a))
check("window default 180 s", S.BACKGROUND_WINDOW_SECS == 180.0, S.BACKGROUND_WINDOW_SECS)
for value, expected in (("", 180.0), ("180s", 180.0), ("abc", 180.0), ("45", 45.0), ("-5", 0.0), ("2.5", 2.5)):
    os.environ["FARABI_TEST_S"] = value
    check("environment variable %r → %s s (a bad value does not crash the engine)" % (value, expected), S._env_secs("FARABI_TEST_S", 180.0) == expected, S._env_secs("FARABI_TEST_S", 180.0))
os.environ.pop("FARABI_TEST_S", None)
check("default when the environment variable is unset", S._env_secs("FARABI_TEST_S", 180.0) == 180.0)

# ── JobQueue.has_family ──
q = S.JobQueue()
check("no family in an empty queue", not q.has_family("qwen") and not q.has_family("gemma"))
q.put(lambda: None, S.BACKGROUND, "qwen"); q.put(lambda: None, S.FRONT, "embed")
check("a Qwen background job is queued, Gemma is not", q.has_family("qwen") and not q.has_family("gemma"))
q.put(lambda: None, S.NORMAL)
check("a load / unload job (no family) does not count as a family", not q.has_family("gemma"))

# ── _release_other ──
def worker(**kw):
    unloaded = []
    d = dict(engine=object(), gemma=object(), paused=None, job_queue=S.JobQueue(), last_external={"qwen": 0.0, "gemma": 0.0}, model_switches=0, unloaded=unloaded)
    d.update(kw)
    i = types.SimpleNamespace(**d)
    i.unload = lambda reason="": (unloaded.append(("qwen", reason)), setattr(i, "engine", None))
    i.unload_gemma = lambda reason="": (unloaded.append(("gemma", reason)), setattr(i, "gemma", None))
    return i
release = S.Worker._release_other

i = worker(gemma=None)
check("Qwen chat, Gemma not loaded: nothing is done", release(i, "qwen") is False and i.unloaded == [])
i = worker()
check("switched to Gemma, Qwen idle (no external request at all): Qwen is unloaded", release(i, "gemma") is True and [b[0] for b in i.unloaded] == ["qwen"] and i.engine is None and i.gemma is not None and i.model_switches == 1)
i = worker()
check("switched to Qwen, Gemma idle: Gemma is unloaded", release(i, "qwen") is True and [b[0] for b in i.unloaded] == ["gemma"] and i.gemma is None and i.engine is not None)
i = worker(paused="qwen")
check("Qwen in a paused background job: not unloaded", release(i, "gemma") is False and i.unloaded == [] and "paused" in logs[-1], logs[-1:])
i = worker(paused="gemma")
check("if the paused job is in THIS family, the other (idle) family is unloaded", release(i, "gemma") is True and [b[0] for b in i.unloaded] == ["qwen"])
i = worker(); i.job_queue.put(lambda: None, S.BACKGROUND, "qwen")
check("a Qwen job is waiting in the queue: not unloaded", release(i, "gemma") is False and i.unloaded == [] and "queued" in logs[-1])
i = worker(); i.job_queue.put(lambda: None, S.FRONT, "gemma"); i.job_queue.put(lambda: None, S.FRONT, "embed")
check("only this family's / embedding jobs are in the queue: the other family is unloaded", release(i, "gemma") is True)
i = worker(last_external={"qwen": time.time() - 30, "gemma": 0.0})
check("Qwen received an external request 30 s ago: not unloaded (in use in the background)", release(i, "gemma") is False and i.unloaded == [] and "not from the app" in logs[-1])
i = worker(last_external={"qwen": time.time() - S.BACKGROUND_WINDOW_SECS - 5, "gemma": time.time()})
check("Qwen's last external request is older than the window: unloaded (Gemma's fresh request is no obstacle)", release(i, "gemma") is True)
i = worker(last_external={"qwen": 0.0, "gemma": time.time() - 10})
check("Gemma is in use by a batch round (10 s ago), a chat switched to Qwen: Gemma is not unloaded", release(i, "qwen") is False and i.unloaded == [])

# ── Worker.chat: flags ──
def chat_worker():
    i = worker()
    i.call = []
    i.current = None
    i._make_halt = lambda cancel, background, family, inter: (lambda: False)
    i._release_other = lambda family: i.call.append(("release", family, dict(i.last_external)))
    i._chat_qwen = lambda *a: i.call.append(("qwen",))
    i.chat_gemma = lambda *a: i.call.append(("gemma",))
    return i
i = chat_worker(); t0 = time.time()
S.Worker.chat(i, {"model": "qwen3.8-27b"}, None, None)
check("unflagged Qwen request (batch pipeline): Qwen gets an 'external use' stamp, no unload is attempted", i.last_external["qwen"] >= t0 and i.last_external["gemma"] == 0.0 and i.call == [("qwen",)])
i = chat_worker()
S.Worker.chat(i, {"model": "gemma4-31b", "client": "app"}, None, None)
check("the app's request without the flag: no stamp, no unload attempted", i.last_external == {"qwen": 0.0, "gemma": 0.0} and i.call == [("gemma",)])
i = chat_worker()
S.Worker.chat(i, {"model": "gemma4-31b", "client": "app", "release_other": True}, None, None)
check("the app's flagged Gemma request: first the decision for the other family, then the chat; no stamp", [c[0] for c in i.call] == ["release", "gemma"] and i.call[0][1] == "gemma" and i.last_external == {"qwen": 0.0, "gemma": 0.0}, i.call)
i = chat_worker()
S.Worker.chat(i, {"model": "qwen3.8-27b", "client": "app", "release_other": True}, None, None)
check("the app's flagged Qwen request: the decision is requested for the Qwen family", i.call[0][:2] == ("release", "qwen"))
i = chat_worker()
def blows_up(*a): raise RuntimeError("x")
i._chat_qwen = blows_up
try:
    S.Worker.chat(i, {"model": "qwen3.8-27b"}, None, None); raised = False
except RuntimeError:
    raised = True
check("the finish stamp is written even when an unflagged request fails", raised and i.last_external["qwen"] > 0)

# ── source check: the rule is triggered only by the flag ──
import inspect
g = inspect.getsource(S.Worker.chat)
check("chat(): the unload decision is made only with the 'release_other' flag", g.count("_release_other(") == 1 and 'body.get("release_other")' in g)
check("version 0.4.5", S.VERSION == "0.4.5", S.VERSION)
print("RESULT: %d / %d" % (elapsed, total))
sys.exit(0 if elapsed == total else 1)
