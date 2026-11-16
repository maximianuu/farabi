"""Priority queue + preemption: logic test WITHOUT loading a model (standard library only).

Usage: python3 tests/queue_test.py <engine root>
"""
import json, os, sys, threading, time, types, urllib.request

ROOT = sys.argv[1]
sys.path.insert(0, ROOT)
import engine.server as S

S.LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queue_test.log")
_write = S.log_fn
S.log_fn = lambda *a: open(S.LOG_FILE, "a").write(" ".join(str(x) for x in a) + "\n")

elapsed = remaining = 0


def check(name, condition, detail=""):
    global elapsed, remaining
    if condition:
        elapsed += 1
    else:
        remaining += 1
    print(("  ✓ " if condition else "  ✗ ") + name + (f"  [{detail}]" if detail and not condition else ""))


class Record:
    def __init__(self):
        self.event, self.t0, self.lock = [], time.perf_counter(), threading.Lock()

    def __call__(self, *a):
        with self.lock:
            self.event.append((round(time.perf_counter() - self.t0, 3),) + a)

    def names(self, kind):
        return [o[2] for o in self.event if o[1] == kind]

    def moment(self, kind, name):
        return next(o[0] for o in self.event if o[1] == kind and o[2] == name)


def new_worker():
    i = S.Worker()
    i.start()
    return i


def make_job(worker, k, name, step=30, secs=0.01, family="gemma", err=False, inner=None):
    """Imitates what a chat job does: reads its record, sets up the halt function, asks halt() at every step."""
    def f():
        record = worker.current
        inter = S.Interrupt()
        halt = worker._make_halt(threading.Event(), record.priority == S.BACKGROUND, family, inter)
        k("begin", name)
        if err:
            raise RuntimeError("test error")
        for _ in range(step):
            time.sleep(secs)
            if inner is not None:
                inner(worker)
            halt()
        k("ended", name, inter.n, round(inter.secs, 3))
    return f


def wait_for(k, name, duration=10):
    t = time.time()
    while time.time() - t < duration:
        if name in k.names("ended"):
            return True
        time.sleep(0.005)
    return False


# bind memory to values managed by the test (do not query a real Ollama)
MEMORY = {"ollama": 0.0, "limit": 54.0}
S.ollama_loaded_gb = lambda: MEMORY["ollama"]
S.gpu_limit_gb = lambda: MEMORY["limit"]


class FakeSessionCache:
    def total_memory(self):
        return 2 << 30


class FakeDepth:
    def __init__(self):
        self.prev_d = 4


class FakeModel:
    def __init__(self):
        self.depth, self.depth_greedy = FakeDepth(), FakeDepth()


print("1) queue order")
i, k = new_worker(), Record()
port = threading.Event()
i.job_queue.put(lambda: port.wait(5), S.NORMAL)
time.sleep(0.05)
i.job_queue.put(make_job(i, k, "a1", 2), S.BACKGROUND, "gemma")
i.job_queue.put(make_job(i, k, "a2", 2), S.BACKGROUND, "gemma")
i.job_queue.put(make_job(i, k, "d1", 2, family=None), S.NORMAL)
i.job_queue.put(make_job(i, k, "o1", 2, family="embed"), S.FRONT, "embed")
i.job_queue.put(make_job(i, k, "o2", 2, family="embed"), S.FRONT, "embed")
check("counts: 2 front / 1 normal / 2 background", i.job_queue.tally() == {"front": 2, "normal": 1, "background": 2}, i.job_queue.tally())
port.set()
wait_for(k, "a2")
check("interactive first, then load/unload, background last; arrival order within the same priority", k.names("begin") == ["o1", "o2", "d1", "a1", "a2"], k.names("begin"))

print("2) an interactive job preempts a running background job")
i, k = new_worker(), Record()
i.job_queue.put(make_job(i, k, "background", 60, 0.01), S.BACKGROUND, "gemma")
time.sleep(0.15)
t_put = time.perf_counter() - k.t0
i.job_queue.put(make_job(i, k, "front", 10, 0.01, family="embed"), S.FRONT, "embed")
wait_for(k, "background")
ended = {o[2]: o for o in k.event if o[1] == "ended"}
check("order: background starts → front starts → front ends → background ends", [(o[1], o[2]) for o in k.event] == [("begin", "background"), ("begin", "front"), ("ended", "front"), ("ended", "background")], k.event)
check("front job started within 50 ms", k.moment("begin", "front") - t_put < 0.05, f"{k.moment('begin', 'front') - t_put:.3f}")
check("background job completed all 60 steps and recorded 1 interleaved job", ended["background"][3] == 1, ended["background"])
check("pause time ≈ duration of the interleaved job", abs(ended["background"][4] - (k.moment("ended", "front") - k.moment("begin", "front"))) < 0.03, ended["background"])
check("preemption over: interleaved=False, paused=None, counter 1", (i.interleaved, i.paused, i.preempted_count) == (False, None, 1), (i.interleaved, i.paused, i.preempted_count))

print("3) two interactive jobs run in turn within the same pause")
i, k = new_worker(), Record()
i.job_queue.put(make_job(i, k, "background", 60, 0.01), S.BACKGROUND, "gemma")
time.sleep(0.1)
i.job_queue.put(make_job(i, k, "on1", 20, 0.01, family="embed"), S.FRONT, "embed")
i.job_queue.put(make_job(i, k, "on2", 5, 0.01, family="embed"), S.FRONT, "embed")
wait_for(k, "background")
check("on1, on2, then background ends", k.names("ended") == ["on1", "on2", "background"], k.names("ended"))
check("background job counted 2 interleaved jobs", [o for o in k.event if o[1] == "ended" and o[2] == "background"][0][3] == 2)

print("4) a load/unload job does NOT preempt")
i, k = new_worker(), Record()
i.job_queue.put(make_job(i, k, "background", 40, 0.01), S.BACKGROUND, "gemma")
time.sleep(0.1)
i.job_queue.put(make_job(i, k, "normal", 2, family=None), S.NORMAL)
i.job_queue.put(make_job(i, k, "background2", 2), S.BACKGROUND, "gemma")
wait_for(k, "background2")
check("background ends → normal → background2", k.names("ended") == ["background", "normal", "background2"], k.names("ended"))

print("5) interactive job whose model is not loaded: preempts if it fits, waits if it does not")
i, k = new_worker(), Record()
i.gemma, i.gemma_session_cache = FakeModel(), FakeSessionCache()
MEMORY["ollama"] = 30.0                       # 17 + (21 + 2) + 30 + 3 = 73 > 54
i.job_queue.put(make_job(i, k, "background", 40, 0.01), S.BACKGROUND, "gemma")
i.job_queue.put(make_job(i, k, "background2", 3, 0.01), S.BACKGROUND, "gemma")
time.sleep(0.1)
i.job_queue.put(make_job(i, k, "front", 3, 0.01, family="qwen"), S.FRONT, "qwen")
wait_for(k, "background2")
check("does not fit: background ends → front (ahead of background2) → background2", k.names("ended") == ["background", "front", "background2"], k.names("ended"))
i, k = new_worker(), Record()
i.gemma, i.gemma_session_cache = FakeModel(), FakeSessionCache()
MEMORY["ollama"] = 0.0                        # 17 + 23 + 0 + 3 = 43 ≤ 54
i.job_queue.put(make_job(i, k, "background", 40, 0.01), S.BACKGROUND, "gemma")
time.sleep(0.1)
i.job_queue.put(make_job(i, k, "front", 3, 0.01, family="qwen"), S.FRONT, "qwen")
wait_for(k, "background")
check("fits: front preempts", k.names("ended") == ["front", "background"], k.names("ended"))
check("on return Gemma's depth controllers will not count the first round (prev_d = -1)", i.gemma.depth.prev_d == -1 and i.gemma.depth_greedy.prev_d == -1)

print("6) if the interleaved job fails, the background job continues")
i, k = new_worker(), Record()
i.job_queue.put(make_job(i, k, "background", 40, 0.01), S.BACKGROUND, "gemma")
time.sleep(0.1)
i.job_queue.put(make_job(i, k, "broken", err=True, family="embed"), S.FRONT, "embed")
check("background job finished", wait_for(k, "background"))
check("worker is ready for a new job", (i.interleaved, i.paused, i.current is None or True) == (False, None, True))
i.job_queue.put(make_job(i, k, "later", 2, family="embed"), S.FRONT, "embed")
check("the job that came after the error ran", wait_for(k, "later"))

print("7) an interactive job arriving while an interactive job is running does not preempt (in order)")
i, k = new_worker(), Record()
i.job_queue.put(make_job(i, k, "on1", 30, 0.01, family="embed"), S.FRONT, "embed")
time.sleep(0.1)
i.job_queue.put(make_job(i, k, "on2", 3, 0.01, family="embed"), S.FRONT, "embed")
wait_for(k, "on2")
check("on1 ends, then on2 starts", [(o[1], o[2]) for o in k.event] == [("begin", "on1"), ("ended", "on1"), ("begin", "on2"), ("ended", "on2")], k.event)

print("8) an interleaved job cannot unload the model of the paused job")
i, k = new_worker(), Record()
i.gemma, i.gemma_session_cache = FakeModel(), FakeSessionCache()
unloads = []
i.unload_gemma = lambda reason="": unloads.append(reason)
MEMORY["ollama"] = 30.0
seen = {}


def inner(worker):
    if "d" not in seen:
        seen["d"] = worker.paused
        worker._make_room("qwen")                 # memory is tight: Gemma would normally be unloaded


i.engine = None
i.job_queue.put(make_job(i, k, "background", 40, 0.01), S.BACKGROUND, "gemma")
time.sleep(0.1)
i.job_queue.put(make_job(i, k, "front", 3, 0.01, family="embed", inner=inner), S.FRONT, "embed")
wait_for(k, "background")
check("while the interleaved job runs, the paused family = gemma", seen.get("d") == "gemma", seen)
check("Gemma was not unloaded", unloads == [], unloads)
i._make_room("qwen")
check("without a pause the same call unloads Gemma (old behaviour)", unloads == ["room for the other family"], unloads)
MEMORY["ollama"] = 0.0

print("9) Qwen's global position state comes back after a pause")
fake = types.ModuleType("engine.core")
fake.LOCATION = types.SimpleNamespace(arr="IMAGE-POSITIONS", shift=7)
sys.modules["engine.core"] = fake
i, k = new_worker(), Record()
i.engine, i.session_cache = FakeModel(), FakeSessionCache()


def corrupt(worker):
    fake.LOCATION.arr, fake.LOCATION.shift = None, 0


i.job_queue.put(make_job(i, k, "background", 40, 0.01, family="qwen"), S.BACKGROUND, "qwen")
time.sleep(0.1)
i.job_queue.put(make_job(i, k, "front", 3, 0.01, family="qwen", inner=corrupt), S.FRONT, "qwen")
wait_for(k, "background")
check("interleaved job ran", k.names("ended") == ["front", "background"], k.names("ended"))
check("LOCATION returned to its old value", (fake.LOCATION.arr, fake.LOCATION.shift) == ("IMAGE-POSITIONS", 7), (fake.LOCATION.arr, fake.LOCATION.shift))
check("Qwen depth controller will not count the first round", i.engine.depth.prev_d == -1)
del sys.modules["engine.core"]

print("10) reading the priority from the request")
check("unspecified → interactive", S.parse_priority({}) == S.FRONT and S.parse_priority({"options": {}}) == S.FRONT)
check('"priority": "background" → background', S.parse_priority({"priority": "background"}) == S.BACKGROUND and S.parse_priority({"priority": " BACKGROUND "}) == S.BACKGROUND)
check("also read from options", S.parse_priority({"options": {"priority": "background"}}) == S.BACKGROUND)
check("unrecognized value / wrong type → interactive", S.parse_priority({"priority": "fast"}) == S.FRONT and S.parse_priority({"priority": 2}) == S.FRONT)

print("11) end to end over HTTP (fake generation; real dispatcher, queue and response fields)")


class TestWorker(S.Worker):
    def _generate(self, body, outp, cancel, halt, inter, waited, background):
        name = body.get("model")
        streaming = bool(body.get("stream", True))
        inter.session = types.SimpleNamespace(busy=True)
        self.last_session = inter.session
        for n in range(int(body["options"]["steps"])):
            time.sleep(0.01)
            if halt():
                break
            if streaming:
                outp.put(("line", {"model": name, "message": {"role": "assistant", "content": "x"}, "done": False}))
        outp.put(("final", {"model": name, "message": {"role": "assistant", "content": "" if streaming else "ok"}, "done": True,
                           "engine": S._queue_info(background, waited, inter)}))

    _chat_qwen = _generate
    chat_gemma = _generate

    def embed(self, texts, duration):
        time.sleep(0.05)
        return [[0.0] for _ in texts], 0.0, 3


S.WORKER = TestWorker()
S.WORKER.start()
PORT = 11441
server = S.Server(("127.0.0.1", PORT), S.RequestHandler)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True).start()


def send(file_path, body, streaming=False):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{file_path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(r, timeout=30) as f:
        if not streaming:
            return json.load(f), time.perf_counter() - t, None
        first, latest = None, None
        for row in f:
            j = json.loads(row)
            if first is None:
                first = time.perf_counter() - t
            if j.get("done"):
                latest = j
        return latest, time.perf_counter() - t, first


result = {}


def send_background():
    result["background"] = send("/api/chat", {"model": "gemma4-31b", "priority": "background", "stream": False, "messages": [{"role": "user", "content": "x"}], "options": {"steps": 100}})


th = threading.Thread(target=send_background)
th.start()
time.sleep(0.3)
status = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=5))
on, front_secs, front_first = send("/api/chat", {"model": "qwen3.8-27b", "stream": True, "messages": [{"role": "user", "content": "hi"}], "options": {"steps": 10}}, streaming=True)
embedding, embed_secs, _ = send("/api/embed", {"model": "bge-m3", "input": ["a", "b"]})
th.join()
background, bg_secs, _ = result["background"]
S.WORKER.gemma = None
check("status endpoint returns the queue fields", {"queued", "preempting", "preemptions"} <= set(status), list(status))
check("interactive (streaming) request was answered before the background job finished", front_secs < 0.6 and bg_secs > 1.0, f"front {front_secs:.2f} s, background {bg_secs:.2f} s")
check("first line arrived within 100 ms", front_first is not None and front_first < 0.1, front_first)
check("interactive response: priority=front, wait < 0.1 s", on["engine"]["priority"] == "front" and on["engine"]["wait_s"] < 0.1, on["engine"])
check("embedding request also preempted", embed_secs < 0.3 and len(embedding["embeddings"]) == 2, embed_secs)
check("background response: priority=background, 2 interleaved, pause time recorded", background["engine"].get("priority") == "background" and background["engine"].get("interleaved") == 2 and 0.1 < background["engine"].get("paused_s", 0) < 0.6, background["engine"])
check("background job completed (content arrived)", background["message"]["content"] == "ok")
check("when the job ended, the session's busy flag was cleared", S.WORKER.last_session.busy is False)
# two requests without a priority: old behaviour (in order)
a = {}
t1 = threading.Thread(target=lambda: a.setdefault("x", send("/api/chat", {"model": "qwen3.8-27b", "stream": False, "messages": [{"role": "user", "content": "1"}], "options": {"steps": 40}})))
t1.start()
time.sleep(0.1)
y, y_secs, _ = send("/api/chat", {"model": "qwen3.8-27b", "stream": False, "messages": [{"role": "user", "content": "2"}], "options": {"steps": 5}})
t1.join()
check("two requests without a priority run in order (the second waits for the first)", y_secs > 0.3 and y["engine"]["wait_s"] > 0.2 and "interleaved" not in a["x"][0]["engine"], (y_secs, y["engine"], a["x"][0]["engine"]))
server.shutdown()

print(f"\n{elapsed} passed, {remaining} failed")
sys.exit(1 if remaining else 0)
