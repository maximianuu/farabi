"""Farabi engine server — the subset of the Ollama API used by the client app.

Endpoints: POST /api/chat (streaming / non-streaming), POST /api/generate (load / unload),
POST /api/embed and /api/embeddings (bge-m3), GET /api/tags, GET /api/ps, POST /api/show, GET /api/version.

Listens only on 127.0.0.1; opens no connection to the outside (HF_HUB_OFFLINE=1).
The model lives on a single worker thread (MLX streams are bound to threads);
requests enter a queue. The queue is prioritized: a chat request whose body has "priority": "background"
(batch jobs from a pipeline) is a background job; if an interactive request arrives while it runs, it stops between
token rounds, the interactive request is interleaved, then the background job continues where it left off (JobQueue, Worker._preempt).

Run:  python -m engine.server [--port 11438]
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import datetime
import errno
import gc
import heapq
import json
import queue
import select
import socket
import socketserver
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

MODEL_NAME = "qwen3.8-27b"
ROOT = os.environ.get("FARABI_MODELS") or os.path.expanduser("~/.farabi/models")
# 2026-11-10: the model quantized here from Qwen's original weights (sha256 verified); the stock conversion (…-nvfp4) is the fallback
MODEL_PATH = os.environ.get("FARABI_MODEL", f"{ROOT}/qwen3.8-27b-nvfp4-gptq")      # 2026-11-11: calibrated (GPTQ) nvfp4 (tools/convert4.py); the previous conversion is kept as a fallback
MTP_PATH = os.environ.get("FARABI_MTP", f"{ROOT}/qwen3.8-27b-mtp/mtp.safetensors")
# 2026-11-11: second model family — Gemma 4 31B (engine/gemma.py; the official weights + the official draft model). If the model name in the request
# is "gemma…" Gemma runs, otherwise Qwen; the two load and unload separately, a single worker runs them in turn.
GEMMA_NAME = "gemma4-31b"
# Model: output of our own converter (tools/gemma_convert3.py). The library's stock conversion is NOT used: for structured text generation it wrote
# `<strong la="…">` instead of `<strong>` (2026-11-11, 8 of 10 outputs were rejected by the format check).
# The embedding table is also 4-bit with our own encoder (tools/gemma_embed.py): same fidelity (KL 0.030 → 0.032), generation 10-38% faster (measured).
GEMMA_PATH = os.environ.get("FARABI_GEMMA", f"{ROOT}/gemma-4-31b-it-nvfp4")
# Draft model: the official Gemma4Assistant, 4-bit with our own encoder (tools/gemma_draft.py; 0.94 → 0.26 GB, step time is halved)
GEMMA_DRAFT = os.environ.get("FARABI_GEMMA_DRAFT", f"{ROOT}/gemma-4-31b-it-assistant-nvfp4")
LEARN_SHORT = 24          # if the checkpoint is within the first 512 tokens its snapshot is at most ≈ 40 MB → even a 24-token gain (≈ 30 ms) is worth it (template prompts)
LEARN_MIN = 192        # minimum shared prefix (tokens) for a learned checkpoint: 192 tokens ≈ 0.25 s of prefill; the snapshot copy is made after the reply
# This many seconds after a model unloads, the tokenizer + image processor kept in the process are released too (Gemma: gemma._PERSISTENT, ≈ 1.1 GB RAM;
# Qwen: core._PERSISTENT + vision._PROCESSOR, ≈ 340 MB; rebuilding takes ≈ 1.9 s). Jobs that load and unload back to back
# (editing instructions, end-of-batch unloads, returning to chat) stay fast within this window.
GEMMA_PERSIST_SECS = float(os.environ.get("FARABI_GEMMA_PERSIST_S", "1800"))
SIZE_GB = {"qwen": 17, "gemma": 21}          # approximate memory (weights + working set); /api/ps reports it (read by the memory guard of external batch pipelines)
LOG_FILE = os.environ.get("FARABI_LOG") or os.path.expanduser("~/Library/Logs/farabi-engine.log")      # environment variable: so a test server does not write to the live log
VERSION = "0.4.5"       # 2026-11-14: when the model changes, the other family is unloaded unless it is used in the background ("release_other"; Worker._release_other); 0.4.4 2026-11-14: MEMORY — root checkpoints thinned (root_checkpoints), session-pool budgets split across both families (pool_budgets), scratch memory released on the cancel / error path, buffer cleared every 64 rounds in generation (session.generate), cap on writing a huge prefix to disk; 0.4.3 2026-11-13: for Gemma schema replies, draft text after the closing no longer enters the reply (Constraint.consume_and_cut); 0.4.2 2026-11-12: object with `properties` in a schema is closed by default (like Ollama; constraint.close_schema) + Qwen's tokenizer is also kept in the process; 0.4.1: Gemma tokenizer kept; 0.4.0: priority queue
WINDOW = 262144
MODEL_IDENT = os.path.basename(MODEL_PATH.rstrip("/"))      # folder name of the on-disk prefix cache (a different folder when the model/conversion changes)
MID_CHECKPOINT = 512       # one snapshot every this many tokens in the system block: prompts that differ at the end re-read at most this much
# 2026-11-14 (0.4.4; context memory use was far too high): MID_CHECKPOINT now applies only at the END of the first block.
# Measurement (engine log, Nov 11-14): for Qwen the median session longer than 6k tokens takes 2.9 GB, of which only ≈ 1.1 GB is attention memory
# and ≈ 1.8 GB is snapshots (≈ 151 MB each, root checkpoints are not pruned). Use of the 512-token checkpoints (Nov 12-14, 460 forks / rewinds):
# 82% were near the end of the block in template prompts shorter than 3.5k tokens; checkpoints in the MIDDLE of a 9-16k system prompt were used ≈ 60 times in three days.
# New layout: one checkpoint every SPARSE_CHECKPOINT in the middle (same as the ones written to disk: disk.INTERVAL), one every MID_CHECKPOINT in the last TAIL_DENSE tokens.
# For a 9.3k-token system prompt: 18 → 10 checkpoints (≈ −1.2 GB); no change for short prompts (≤ 3.5k).
# The 512-token checkpoints also stay within the first SPARSE_CHECKPOINT tokens (review finding): template prompts with a 1-2k-token SHARED header
# (a fixed rule header plus a varying body) are not re-read from the start even when the block is longer than 2k. Costs at most 3 checkpoints (≈ 450 MB).
SPARSE_CHECKPOINT = 2048   # one checkpoint at this interval in the middle of the first block; must stay the SAME as disk.INTERVAL (disk._chosen writes these multiples)
TAIL_DENSE = 1536     # one checkpoint every MID_CHECKPOINT in the last this-many tokens of the first block (the client puts varying blocks at the end; template prompts also diverge here)
ROOT_CAP = 16384       # sparse checkpoints only up to this position: in a single huge message with no system message (first block = whole prompt) the number of checkpoints
                      # stays bounded. Measured: one 96k-token message held 186 checkpoints = 33.7 GB and the process reached 54 GB; now at most 19 checkpoints.
# Session pool budgets (GB). Old layout: Qwen fixed at 12 GB, only Gemma's shrank depending on Qwen → both families together 17 + 12 + 21 + 3..10
# = 53-60 GB against a GPU limit of 54 GB (it fell into swap more than once). Now the free space is split between the two pools in proportion to their caps.
POOL_CAP = {"qwen": 12.0, "gemma": 10.0}     # pool cap when the family is loaded alone (the old constants)
POOL_FLOOR = {"qwen": 3.0, "gemma": 3.0}     # at least this much (root + one or two conversations)
POOL_SHARE = 8.0                                # margin left for the prefill peak + other GPU users (the 8 in the old Gemma formula)
# 0.4.5 (2026-11-14): when the main chat request from the app carries "release_other": true, the other family, if loaded and not used in the background,
# is unloaded before the request starts. "Used in the background" = has a paused job · has a queued job · its last request NOT from the app
# (one without "client": "app") is newer than BACKGROUND_WINDOW_SECS. The priority flag ("background") is not consulted: batch pipelines send
# Qwen requests with images at front priority, and tool connectors send them without a flag (all of these count as "not from the app").
# The window comes from measurement (engine log, Nov 12-14): within a batch run, the gap between consecutive background requests of the same family is,
# at the 99th percentile, 125 s for Qwen and 25 s for Gemma; gaps longer than 180 s (≤ 10 min) occurred 45 times in three days. Had the rule fired on every
# interactive request, the family would have been requested again within 10 min after 58 of 61 unloads → the only trigger is the app's explicit flag.
def _env_secs(name: str, fallback: float) -> float:
    """Environment variable in seconds; an empty / malformed value must not crash the engine at startup (it falls back to the default), a negative value becomes 0."""
    try:
        return max(0.0, float(os.environ.get(name) or fallback))
    except (TypeError, ValueError):
        return float(fallback)


BACKGROUND_WINDOW_SECS = _env_secs("FARABI_BACKGROUND_WINDOW_S", 180.0)
KV_KB = {"qwen": 72.0, "gemma": 82.0}          # context memory per prefilled token (KB; measured 2026-11-14, ≈ 102k-token prompt)


def root_checkpoints(k: int) -> set:
    """Snapshot locations of the first block; k = position of the second <|im_start|> (the end of the first block). k inclusive; empty if k < MID_CHECKPOINT.
    The head (≤ SPARSE_CHECKPOINT) and the tail (TAIL_DENSE) get one every MID_CHECKPOINT, the middle one every SPARSE_CHECKPOINT (up to ROOT_CAP), then one every 2 × SPARSE_CHECKPOINT
    (up to 2 × ROOT_CAP: in a 16-32k-token system prompt a change in the middle forces re-reading at most 4,095 tokens; at most +4 checkpoints)."""
    if k < MID_CHECKPOINT:
        return set()
    place = {k}
    for x in range(MID_CHECKPOINT, k, MID_CHECKPOINT):
        if (x >= k - TAIL_DENSE or x <= SPARSE_CHECKPOINT or (x % SPARSE_CHECKPOINT == 0 and x <= ROOT_CAP)
                or (x % (2 * SPARSE_CHECKPOINT) == 0 and x <= 2 * ROOT_CAP)):
            place.add(x)
    return place


def pool_budgets(loaded, limit: float, ollama: float, usage: Optional[dict] = None) -> dict:
    """Session pool budgets (GB) of the loaded families. Free space = limit − weights − Ollama − margin. Each family's GUARANTEED share is its
    portion of the free space in proportion to the caps; space the other family is not USING at that moment is also opened to it (usage: GB currently held in the families' pools),
    clamped to [floor, cap]. Example (54 GB limit, 8 GB free): with both pools full Qwen ≈ 4.4 + Gemma ≈ 3.6; with Qwen's pool empty Gemma gets 8
    (same as the 0.4.3 Gemma formula); with a single family loaded, the old constants (Qwen 12, Gemma 10).
    As one family grows, the other's budget is recomputed at the end of the request and the excess is pruned (Worker._apply_budget)."""
    loaded = [a for a in loaded if a in POOL_CAP]
    usage = usage or {}
    free_mem = limit - sum(SIZE_GB[a] for a in loaded) - ollama - POOL_SHARE
    total_cap = sum(POOL_CAP[a] for a in loaded) or 1.0
    b = {}
    for a in loaded:
        margin = free_mem * POOL_CAP[a] / total_cap
        others = sum(usage.get(d, 0.0) for d in loaded if d != a)
        b[a] = max(POOL_FLOOR[a], min(POOL_CAP[a], max(margin, free_mem - others)))
    return b
DISK_GAIN = 1024    # the disk prefix is preferred if it is at least this much longer than the in-memory one (restore ≈ 0.1-0.6 s; reading 1,024 tokens ≈ 1.1 s)
DEFAULTS = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}   # the model defaults in Ollama


PROFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network-lock.sb")
NET_LOCK = False        # measured at startup: can the process open connections outside this Mac


def net_lock_setup():
    """Restarts the process inside a macOS sandbox: it can talk only to 127.0.0.1 (engine/network-lock.sb).

    The model files and libraries were downloaded from third parties; so that nothing has to be trusted, the lock is
    enforced by the operating system. Skipped with FARABI_NETWORK_LOCK=off (for debugging only).
    """
    status = os.environ.get("FARABI_NETWORK_LOCK")
    if status in ("1", "off"):
        return
    if not (os.path.exists("/usr/bin/sandbox-exec") and os.path.exists(PROFILE)):
        raise SystemExit("network lock could not be set up: sandbox-exec or network-lock.sb is missing (skip with FARABI_NETWORK_LOCK=off)")
    os.environ["FARABI_NETWORK_LOCK"] = "1"
    os.execv("/usr/bin/sandbox-exec", ["/usr/bin/sandbox-exec", "-f", PROFILE, sys.executable, "-m", "engine.server"] + sys.argv[1:])


def net_lock_active() -> bool:
    """Does the operating system refuse an outbound connection attempt (192.0.2.1: a test address that routes nowhere)."""
    sock = socket.socket()
    sock.settimeout(0.4)
    try:
        sock.connect(("192.0.2.1", 9))
        return False
    except PermissionError:
        return True
    except OSError as e:
        return e.errno == errno.EPERM
    finally:
        sock.close()


def is_gemma(name) -> bool:
    return isinstance(name, str) and name.startswith("gemma")


def gemma_installed() -> bool:
    return os.path.isdir(GEMMA_PATH) and os.path.isdir(GEMMA_DRAFT)


_gpu_limit = [0.0, 0.0]      # (value, time read): so sysctl is not called on every request


def gpu_limit_gb() -> float:
    """Memory the GPU can use (iogpu.wired_limit_mb; 75% of memory if unset). Re-read every 10 minutes."""
    if _gpu_limit[0] and time.time() - _gpu_limit[1] < 600:
        return _gpu_limit[0]
    _gpu_limit[0], _gpu_limit[1] = _read_gpu_limit(), time.time()
    return _gpu_limit[0]


def _read_gpu_limit() -> float:
    try:
        mb = int(subprocess.run(["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True, timeout=3).stdout.strip() or 0)
        if mb > 0:
            return mb / 1024
        return int(subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3).stdout.strip()) * 0.75 / (1 << 30)
    except Exception:
        return 48.0


def ollama_loaded_gb() -> float:
    """Models Ollama currently holds in memory (127.0.0.1 only; the lock allows this). 0 if unreachable."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=2) as y:
            return sum(m.get("size_vram", 0) for m in json.load(y).get("models", [])) / (1 << 30)
    except Exception:
        return 0.0


def _timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def log_fn(*a):
    row = datetime.datetime.now().strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(row + "\n")
    except Exception:
        pass
    print(row, flush=True)


def parse_duration(v, fallback=600.0) -> float:
    """Ollama keep_alive: "10m", "1h", 300, -1 (forever), 0 (unload immediately) → seconds."""
    if v is None:
        return fallback
    if isinstance(v, (int, float)):
        return float("inf") if v < 0 else float(v)
    s = str(v).strip().lower()
    try:
        if s.startswith("-"):
            return float("inf")
        for extra, factor in (("ms", 0.001), ("s", 1), ("m", 60), ("h", 3600)):
            if s.endswith(extra):
                return float(s[: -len(extra)]) * factor
        return float(s)
    except Exception:
        return fallback


FRONT, NORMAL, BACKGROUND = 0, 1, 2      # job priorities: interactive chat + embedding · load/unload · background chat (batch jobs from a pipeline)
BACKGROUND_WORDS = ("background", "batch", "low")


def parse_priority(body: dict) -> int:
    """Priority of a chat request: "priority": "background" in the body (or in options) → background; if unspecified, interactive (the old behaviour)."""
    v = body.get("priority")
    if v is None:
        v = (body.get("options") or {}).get("priority")
    return BACKGROUND if isinstance(v, str) and v.strip().lower() in BACKGROUND_WORDS else FRONT


class Job:
    """A queued job: f is called on the worker thread."""

    __slots__ = ("f", "priority", "family", "seq", "t")

    def __init__(self, f, priority: int = NORMAL, family: Optional[str] = None):
        self.f, self.priority, self.family = f, priority, family       # family: "qwen" | "gemma" | "embed" | None (load/unload)
        self.seq, self.t = 0, time.time()


class Interrupt:
    """Pause record of a chat job: the time spent in interleaved jobs (subtracted from the statistics) and the session it was using at that point."""

    __slots__ = ("secs", "n", "session", "read_n")

    def __init__(self):
        self.secs, self.n, self.session, self.read_n = 0.0, 0, None, 0      # read_n: how many of the interleaved jobs came in while a prompt was being read (the rest came during generation)


class JobQueue:
    """The worker's job queue: interactive jobs first (FRONT), then load/unload (NORMAL), background jobs last (BACKGROUND); arrival order within the same priority."""

    def __init__(self):
        self._condition = threading.Condition()
        self._jobs: list = []          # heap of (priority, sequence number, Job)
        self._seq = 0

    def put(self, f, priority: int = NORMAL, family: Optional[str] = None) -> Job:
        job_ = f if isinstance(f, Job) else Job(f, priority, family)
        with self._condition:
            self._seq += 1
            job_.seq = self._seq
            heapq.heappush(self._jobs, (job_.priority, job_.seq, job_))
            self._condition.notify()
        return job_

    def get(self, timeout: float) -> Job:
        with self._condition:
            if not self._jobs:
                self._condition.wait(timeout)
            if not self._jobs:
                raise queue.Empty
            return heapq.heappop(self._jobs)[2]

    def has_front(self) -> bool:
        """Is an interactive job waiting in the queue (a background job asks this on every token round)."""
        with self._condition:
            return bool(self._jobs) and self._jobs[0][0] == FRONT

    def take_front(self, fits) -> Optional[Job]:
        """The interactive job at the head of the queue (if fits(job) is true); None if there is none. Does not wait."""
        with self._condition:
            if self._jobs and self._jobs[0][0] == FRONT and fits(self._jobs[0][2]):
                return heapq.heappop(self._jobs)[2]
        return None

    def has_family(self, family: str) -> bool:
        """Is a job waiting in the queue for this family (priority does not matter)."""
        with self._condition:
            return any(i.family == family for _, _, i in self._jobs)

    def tally(self) -> dict:
        with self._condition:
            return {"front": sum(1 for o, _, _ in self._jobs if o == FRONT), "normal": sum(1 for o, _, _ in self._jobs if o == NORMAL),
                    "background": sum(1 for o, _, _ in self._jobs if o == BACKGROUND)}


class Worker(threading.Thread):
    """The single thread that owns the model."""

    def __init__(self):
        super().__init__(daemon=True)
        self.job_queue = JobQueue()
        self.current: Optional[Job] = None    # the job running right now (while an interleaved job runs, that one)
        self.interleaved = False                   # a background job is paused and the interleaved job is running
        self.paused: Optional[str] = None  # family of the paused job: the interleaving job cannot unload this family
        self.preempted_count = 0                # number of jobs interleaved since startup
        self._fits_at = (0.0, None, True)   # (time, family, decision): so Ollama is not asked the memory question on every round
        self._ollama_at = (0.0, 0.0)         # (time, GB): Ollama's loaded memory for the pool budget, cached for 15 s (0.4.4)
        self.last_external = {"qwen": 0.0, "gemma": 0.0}   # per family, the time of the last chat request NOT from the app (written at start and at finish; 0.4.5)
        self.model_switches = 0              # number of families unloaded via "release_other" since startup
        self._release_pending = False         # the memory of a session discarded on the cancel / error path will be released when the job ends (0.4.4; see _release_memory)
        self.engine = None
        self.session_cache = None
        self.finish = 0.0            # the model is unloaded after this time
        self.busy = False
        self.load_secs = 0.0
        self.constraint_tok = None
        self.embedding = None           # bge-m3 (memory / library searches)
        self.embedding_finish = 0.0
        self.gemma = None           # Gemma 4 31B (engine/gemma.py)
        self.gemma_session_cache = None
        self.gemma_finish = 0.0
        self.gemma_load_secs = 0.0
        self.gemma_unloaded_at = 0.0    # time Gemma left memory (0: no tokenizer is kept / it was released)
        self.qwen_unloaded_at = 0.0     # the same for Qwen (core._PERSISTENT + vision._PROCESSOR)

    def run(self):
        while True:
            try:
                job_ = self.job_queue.get(timeout=5)
            except queue.Empty:
                if self.engine is not None and time.time() > self.finish:
                    self.unload("time expired")
                if self.gemma is not None and time.time() > self.gemma_finish:
                    self.unload_gemma("time expired")
                if self.embedding is not None and time.time() > self.embedding_finish:
                    self.embedding = None
                    gc.collect()
                    log_fn("embedding model unloaded")
                if self.gemma is None and self.gemma_unloaded_at and time.time() - self.gemma_unloaded_at > GEMMA_PERSIST_SECS:
                    self._release_persistent()
                if self.engine is None and self.qwen_unloaded_at and time.time() - self.qwen_unloaded_at > GEMMA_PERSIST_SECS:
                    self._release_persistent_qwen()
                continue
            self.busy = True
            self.current = job_
            try:
                job_.f()
            except Exception:
                log_fn("job error:", traceback.format_exc()[-800:])
            finally:
                self.current = None
                self.busy = False
            if self._release_pending:             # exception and job frames are gone: the arrays of the discarded session are really freed
                self._release_pending = False
                self._release_memory()

    # -- priority: pausing a background job to interleave an interactive one (only from the worker thread)
    def _fits(self, job_: Job) -> bool:
        """Can the job to be interleaved run while the paused job's model is in memory: if its own model is loaded or both fit together under the GPU limit.
        If they do not fit it is not interleaved; it runs from the head of the queue when the background job finishes (then _make_room unloads the other family — the old behaviour)."""
        a = job_.family
        if a not in ("qwen", "gemma"):
            return True                          # the embedding model ≈ 1.2 GB
        if (self.engine if a == "qwen" else self.gemma) is not None:
            return True
        t_now = time.time()
        if self._fits_at[1] == a and t_now - self._fits_at[0] < 5:
            return self._fits_at[2]
        other = "gemma" if a == "qwen" else "qwen"
        other_gb = 0.0
        if (self.gemma if other == "gemma" else self.engine) is not None:
            scache = self.gemma_session_cache if other == "gemma" else self.session_cache
            other_gb = SIZE_GB[other] + (scache.total_memory() / (1 << 30) if scache is not None else 0)
        decision = SIZE_GB[a] + other_gb + ollama_loaded_gb() + 3 <= gpu_limit_gb()
        if decision != self._fits_at[2] or self._fits_at[1] != a:
            log_fn(f"interleave: {a} is not loaded, {other} {other_gb:.1f} GB together with {'fits' if decision else 'DOES NOT FIT — waiting for the background job to finish'}")
        self._fits_at = (t_now, a, decision)
        return decision

    def _preempt(self, family: str, inter: Interrupt):
        """The running BACKGROUND job stops here between token rounds (at the halt() call); waiting interactive jobs run in turn on the same thread,
        then the background job continues where it left off. The paused job's state lives in its own session and local variables; three shared things are
        preserved: (1) its session is marked `busy` → the pool does not reuse / discard it in place (it only forks from its snapshot), (2) its family cannot be unloaded
        (`paused`), (3) Qwen's global location state (LOCATION) is saved and restored. The depth controller does not count the first round after the return.
        The random number generator is global: an interleaved SAMPLING background job will not produce exactly the same text with the same seed (it does at temperature 0)."""
        if self.interleaved:
            return
        job_ = self.job_queue.take_front(self._fits)
        if job_ is None:
            return
        t = time.perf_counter()
        n = 0
        outer = self.current
        location = None
        if self.engine is not None:
            from .core import LOCATION
            location = (LOCATION, LOCATION.arr, LOCATION.shift)
        self.interleaved, self.paused = True, family
        try:
            while job_ is not None:
                self.current = job_
                n += 1
                try:
                    job_.f()
                except Exception:
                    log_fn("job error (interleaved):", traceback.format_exc()[-800:])
                if self._release_pending:
                    self._release_pending = False
                    self._release_memory()
                job_ = self.job_queue.take_front(self._fits)
        finally:
            self.current = outer
            self.interleaved, self.paused = False, None
            if location is not None:
                location[0].arr, location[0].shift = location[1], location[2]
            for d in ([self.engine.depth] if family == "qwen" and self.engine is not None else
                      [self.gemma.depth, self.gemma.depth_greedy] if family == "gemma" and self.gemma is not None else []):
                d.prev_d = -1
            secs = time.perf_counter() - t
            inter.secs += secs
            inter.n += n
            self.preempted_count += n
            log_fn(f"interleaved: {n} job(s) {secs:.1f} s ({family} background job paused, continuing)")

    def _make_halt(self, cancel: threading.Event, background: bool, family: str, inter: Interrupt):
        """The "halt" function that prefill/generation loops ask on every step. In a background job the queue is checked at the same point: if there is an interactive job it is interleaved."""
        if not background:
            return cancel.is_set

        def halt():
            if self.job_queue.has_front():
                self._preempt(family, inter)
            return cancel.is_set()
        return halt

    # -- called only from the worker thread
    def _make_room(self, family: str):
        """Before loading: if two families + Ollama's models would exceed the GPU limit, unloads the idle OTHER family (does not touch Ollama)."""
        other = "gemma" if family == "qwen" else "qwen"
        other_loaded = (self.gemma if other == "gemma" else self.engine) is not None
        if not other_loaded:
            return
        limit, ollama = gpu_limit_gb(), ollama_loaded_gb()
        # the other family's session cache also takes space (Qwen ≤ 12 GB, Gemma ≤ 8 GB); 3 GB margin
        scache = self.gemma_session_cache if other == "gemma" else self.session_cache
        other_gb = SIZE_GB[other] + (scache.total_memory() / (1 << 30) if scache is not None else 0)
        if SIZE_GB[family] + other_gb + ollama + 3 > limit:
            if other == self.paused:
                log_fn(f"cannot make room: {other} is in use by a paused job; loading {family} alongside it")
                return
            log_fn(f"making room: {family} will be loaded; {other} {other_gb:.1f} GB + Ollama {ollama:.1f} GB, limit {limit:.0f} GB")
            (self.unload_gemma if other == "gemma" else self.unload)("room for the other family")

    def _release_other(self, family: str) -> bool:
        """In the client app the user switched to talking with this family ("release_other"): if the OTHER family is loaded and not used in the background it is unloaded
        immediately (the keep_alive time is not waited for); this family's weights are loaded afterwards. Called only from the worker thread, at the start of a chat job."""
        other = "gemma" if family == "qwen" else "qwen"
        if (self.gemma if other == "gemma" else self.engine) is None:
            return False
        elapsed = time.time() - self.last_external.get(other, 0.0)
        reason = ("in use by a paused job" if self.paused == other else
                 "has a queued job" if self.job_queue.has_family(other) else
                 f"last request not from the app {elapsed:.0f} s ago (window {BACKGROUND_WINDOW_SECS:.0f} s)" if elapsed < BACKGROUND_WINDOW_SECS else None)
        if reason is not None:
            log_fn(f"model changed → {family}: {other} not unloaded — {reason}")
            return False
        (self.unload_gemma if other == "gemma" else self.unload)(f"model changed → {family}; not used in the background")
        self.model_switches += 1
        return True

    def _budget_before_read(self, family: str, o, to_read: int):
        """When both families are loaded, BEFORE a long prompt is read, the pool share the other family borrowed is taken back (0.4.4; second review):
        the budget is set as if this session had already grown by the number of tokens to be read. It is the same as the pruning done at the end of a request, only done before the
        prefill peak (no extra cache loss). Not called for small reads (< 0.25 GB ≈ 3.5k tokens) or when a single family is loaded."""
        if self.engine is None or self.gemma is None:
            return
        gb = max(0, to_read) * KV_KB[family] / (1 << 20)
        if gb >= 0.25:
            self._apply_budget(protect=o, growing=(family, gb), poll=False)

    def _apply_budget(self, protect=None, to_load: Optional[str] = None, growing=None, poll: bool = True):
        """Sets the session pools' budgets by the loaded families and prunes the excess (0.4.4). Called at the end of a request (after the reply is sent) and when a family
        is loaded. to_load: the family whose weights are ABOUT to be loaded — the other family's pool shrinks before the new weights are allocated.
        In each pool the most recently used session is not discarded: `protect` if it is in that pool, otherwise that pool's newest session (so that the other family's request
        finishing does not evict this family's warm conversation — review finding). The session of a paused job (busy) is never discarded by prune anyway."""
        pools = {"qwen": self.session_cache if self.engine is not None else None, "gemma": self.gemma_session_cache if self.gemma is not None else None}
        loaded = [a for a, mm in (("qwen", self.engine), ("gemma", self.gemma)) if mm is not None or a == to_load]
        if not loaded:
            return
        t_now = time.time()
        if poll and t_now - self._ollama_at[0] > 15:    # do not ask Ollama on every request; the call before prefill never asks (it must not hold up the reply)
            self._ollama_at = (t_now, ollama_loaded_gb())
        usage = {a: (getattr(pools[a], "total_memory", lambda: 0)() / (1 << 30) if pools.get(a) is not None else 0.0) for a in loaded}
        if growing is not None and growing[0] in usage:      # (family, GB): this family is about to grow by this much (the prompt to be read)
            usage[growing[0]] += growing[1]
        b = pool_budgets(loaded, gpu_limit_gb(), self._ollama_at[1], usage)
        for a in loaded:
            scache = pools.get(a)
            if scache is None or not hasattr(scache, "prune"):
                continue
            new = int(b[a] * (1 << 30))
            old = getattr(scache, "max_bytes", new)
            if abs(old - new) >= (1 << 30):          # the budget moves with usage; only changes of ≥ 1 GB are written to the log
                log_fn(f"pool budget: {a} {old / (1 << 30):.1f} → {b[a]:.1f} GB (loaded: {'+'.join(loaded)}; used "
                       + ", ".join(f"{d} {usage[d]:.1f}" for d in loaded) + " GB)")
            scache.max_bytes = new
            k = protect if protect is not None and any(x is protect for x in scache.sessions) else max(scache.sessions, key=lambda x: x.latest, default=None)
            scache.prune(protect=k)

    @staticmethod
    def _release_memory():
        """The arrays of a session discarded on the cancel / error path wait in MLX's buffer memory; on exits where prune() is not called they are released here (0.4.4).
        Measured: after a cancelled 96k-token read the process stayed at 52 GB until the model was unloaded.
        NOT called while an exception is being handled: the traceback holds the step_forward() frame, which holds the session (the first version was inside the except and had no effect — measured).
        The branches only set `self._release_pending = True`; this runs when the job function returns (run / _preempt)."""
        try:
            import mlx.core as mx

            gc.collect()
            mx.clear_cache()
        except Exception:
            pass

    def load_gemma(self) -> float:
        if self.gemma is not None:
            return 0.0
        self._make_room("gemma")
        self._apply_budget(to_load="gemma")     # 0.4.4: if Qwen stayed loaded, its pool shrinks BEFORE Gemma's weights are allocated
        t = time.time()
        from .gemma import GemmaEngine, GemmaSessionCache

        self.gemma = GemmaEngine(GEMMA_PATH, GEMMA_DRAFT)
        self.gemma_session_cache = GemmaSessionCache(self.gemma)
        self._apply_budget()            # 0.4.4: the budget of the new pool is set
        self.gemma_load_secs = time.time() - t
        log_fn(f"gemma loaded {self.gemma_load_secs:.1f} s" + (" (tokenizer was ready in the process)" if getattr(self.gemma, "stored_processor", False) else ""))
        return self.gemma_load_secs

    def unload_gemma(self, reason=""):
        if self.gemma is None:
            return
        import mlx.core as mx

        self.gemma = None
        self.gemma_session_cache = None
        gc.collect()
        mx.clear_cache()
        self.gemma_unloaded_at = time.time()
        log_fn("gemma unloaded", reason)

    def _release_persistent(self):
        """GEMMA_PERSIST_SECS after Gemma unloads: the tokenizer + image processor kept in the process are released too (≈ 1.1 GB RAM comes back).
        The next load rebuilds them (≈ 1.9 s). mlx is not imported: if Gemma was never loaded the module does not exist either."""
        self.gemma_unloaded_at = 0.0
        g = sys.modules.get((__package__ or "engine") + ".gemma")
        if g is not None and getattr(g, "_PERSISTENT", None):
            g._PERSISTENT.clear()
            gc.collect()
            log_fn("gemma tokenizer released (idle)")

    def _release_persistent_qwen(self):
        """GEMMA_PERSIST_SECS after Qwen unloads: the chat tokenizer + vision processor kept in the process are released (≈ 340 MB RAM)."""
        self.qwen_unloaded_at = 0.0
        released = False
        for name, area in (("core", "_PERSISTENT"), ("vision", "_PROCESSOR")):
            m = sys.modules.get((__package__ or "engine") + "." + name)
            d = getattr(m, area, None) if m is not None else None
            if d:
                d.clear()
                released = True
        if released:
            gc.collect()
            log_fn("qwen tokenizer released (idle)")

    def do_load(self) -> float:
        if self.engine is not None:
            return 0.0
        self._make_room("qwen")
        self._apply_budget(to_load="qwen")      # 0.4.4: if Gemma stayed loaded, its pool shrinks BEFORE Qwen's weights are allocated
        t = time.time()
        from .core import Engine
        from .session import SessionCache

        self.engine = Engine(MODEL_PATH, MTP_PATH, vision=True)
        self.session_cache = SessionCache(self.engine)
        self._apply_budget()            # 0.4.4: the budget of the new pool is set
        self.load_secs = time.time() - t
        log_fn(f"model loaded {self.load_secs:.1f} s" + (" (tokenizer was ready in the process)" if getattr(self.engine, "stored_tok", False) else ""))
        return self.load_secs

    def unload(self, reason=""):
        if self.engine is None:
            return
        import mlx.core as mx

        self.engine = None
        self.session_cache = None
        self.constraint_tok = None
        gc.collect()
        mx.clear_cache()
        self.qwen_unloaded_at = time.time()
        log_fn("model unloaded", reason)

    def embed(self, texts: list, duration: float):
        """bge-m3 embeddings; the model is loaded on the first request (≈1 s) and stays in memory for `duration`."""
        t = time.time()
        load_time = 0.0
        if self.embedding is None:
            from .embedding import Embedder

            self.embedding = Embedder()
            load_time = time.time() - t
            log_fn(f"embedding model loaded {load_time:.1f} s")
        v = self.embedding.embed(texts)
        self.embedding_finish = time.time() + duration
        log_fn(f"embed {len(texts)} texts {self.embedding.last_token} tokens {time.time() - t:.2f} s")
        return v, load_time, self.embedding.last_token

    def chat(self, body: dict, outp: "queue.Queue", cancel: threading.Event):
        record = self.current
        background = record is not None and record.priority == BACKGROUND
        gemma = is_gemma(body.get("model"))
        family = "gemma" if gemma else "qwen"
        from_app = body.get("client") == "app"          # a request from the app itself does not make the family count as "used in the background" (0.4.5)
        if not from_app:
            self.last_external[family] = time.time()
        if body.get("release_other"):
            self._release_other(family)
        inter = Interrupt()
        halt = self._make_halt(cancel, background, family, inter)
        waited = max(0.0, time.time() - record.t) if record is not None else 0.0
        try:
            return (self.chat_gemma if gemma else self._chat_qwen)(body, outp, cancel, halt, inter, waited, background)
        finally:
            if not from_app:
                self.last_external[family] = time.time()   # also when a long job finishes: the window is counted from the END of the job
            if inter.session is not None:
                inter.session.busy = False

    def _chat_qwen(self, body: dict, outp: "queue.Queue", cancel: threading.Event, halt, inter: Interrupt, waited: float, background: bool):
        import mlx.core as mx

        from .session import Cancelled, Stats, step_forward, generate
        from .sampling import Sampler
        from .chat import Parser, flatten, build_prompt

        t_begin = time.time()
        load_time = self.do_load()
        m, scache = self.engine, self.session_cache
        pick = body.get("options") or {}
        tool_list = body.get("tools") or None
        schema = body.get("format")
        think = body.get("think", True)
        if schema:
            think = False                      # schema enforcement is not applied together with a thinking block
        messages, image_list, digests = flatten(body.get("messages") or [])
        prompt = build_prompt(m.tok, messages, tool_list, think)

        gb = None
        if image_list:
            gb = m.vision.to_tokens(prompt, image_list)
            ids, places = gb["ids"], gb["positions"]
        else:
            ids, places = m.tok.encode(prompt, add_special_tokens=False), []
        if len(ids) >= WINDOW:
            raise ValueError(f"prompt {len(ids)} tokens; window {WINDOW}")

        single = bool(schema) or body.get("cache") is False
        o, how = scache.pick(ids, digests, places, transient=single)
        disk_secs, mem_begin = 0.0, len(o.tok)
        if how in ("new", "fork") and gb is None and len(ids) > 1200:
            # 2026-11-12: no matching session in memory (model just loaded / engine restarted) → if the prefix is on disk, restore it instead of reading.
            # Second round (same day, test log): disk is also checked when there is a SHORT match in memory — when the set of loaded tools changes, the prefix
            # shared with in-memory sessions stays at 1-2k tokens ("fork"), while the same set sat on disk with 12-14k tokens and was never looked up
            # (17k tokens ≈ 19 s of re-reading). If the disk prefix is at least DISK_GAIN tokens longer, the fork is dropped and the disk is loaded into an empty session.
            from . import disk
            from .session import Session as _Session
            td = time.perf_counter()
            try:
                dk, file, disk_meta = disk.longest(ids, MODEL_IDENT, o)
                if dk >= mem_begin + DISK_GAIN:
                    if how == "fork":
                        scache.release(o)
                        o = _Session(m)
                        scache.sessions.append(o)
                    disk.do_load(o, ids, MODEL_IDENT, dk, file, disk_meta)
                    how = "disk"
            except Exception as e:
                log_fn(f"prefix cache could not be read from disk ({type(e).__name__}: {e}) — reading from the start")
                scache.release(o)
                o = _Session(m)
                scache.sessions.append(o)
                how = "new"
            disk_secs = time.perf_counter() - td
        begin = len(o.tok)
        image = None
        if gb is not None and any(y >= begin for y in places):
            image = (m.vision.embed(gb, digests, m._embed, begin), gb["position"], gb["offset"])
        # snapshot locations: the end of the system block (if long) and the start of the last generation
        begin_tok = m.tok.convert_tokens_to_ids("<|im_start|>")
        starts = [i for i, b in enumerate(ids) if b == begin_tok]
        place = set()
        if len(starts) >= 2:
            place.add(starts[-1])
            if starts[1] >= 512:
                place.update(root_checkpoints(starts[1]))      # intermediate checkpoints in the first block (2026-11-11); 0.4.4: sparse in the middle, dense at the end, bounded in number
                o.root_limit = max(o.root_limit, starts[1])
        if gb is not None and gb["positions"]:
            # end of the last image: when a different question is asked about the same image (browser steps) the image is not re-read
            place.add(gb["positions"][-1] + gb["parts"][-1][3] + 1)
        stats = Stats(prompt_tok=len(ids), read_tok=len(ids) - begin)
        name = body.get("model") or MODEL_NAME

        def row(message: dict, **extra):
            d = {"model": name, "created_at": _timestamp(),
                 "message": {"role": "assistant", "content": "", **message}, "done": False}
            d.update(extra)
            outp.put(("line", d))

        t = time.perf_counter()
        inter.session, o.busy = o, True     # while the job runs (even if paused) the pool does not reuse / discard this session in place
        self._budget_before_read("qwen", o, len(ids) - begin)
        try:
            step_forward(m, o, ids, image=image, snapshot_places=place, halt=halt)
        except Cancelled:
            scache.release(o)
            inter.session = None
            self._release_pending = True        # the memory is released when the job function returns (the traceback holds the session here)
            outp.put(("final", None))
            log_fn(f"cancelled (while reading) prompt={len(ids)}")
            return
        except Exception:
            scache.release(o)            # a half-read session may be inconsistent
            inter.session = None
            self._release_pending = True
            raise
        o.image_list = list(digests)
        stats.prompt_secs = time.perf_counter() - t - inter.secs      # time spent in interleaved jobs does not count toward the prefill time
        inter_read, inter.read_n = inter.secs, inter.n

        sampler = Sampler(
            temperature=float(pick.get("temperature", DEFAULTS["temperature"])),
            top_k=int(pick.get("top_k", DEFAULTS["top_k"])),
            top_p=float(pick.get("top_p", DEFAULTS["top_p"])),
            min_p=float(pick.get("min_p", DEFAULTS["min_p"])),
            rng_seed=pick.get("seed"),
        )
        max_n = int(pick.get("num_predict", -1) or -1)
        if max_n <= 0:
            max_n = max(16, min(32768, WINDOW - len(ids)))
        constraint = None
        if schema:
            from .constraint import Constraint

            constraint = Constraint(m.tok._tokenizer, {"type": "object"} if schema == "json" else schema)

        trace = [] if pick.get("trace") else None          # diagnostics: the dump of the rounds goes into the response's engine.trace field
        parser = Parser(bool(think), tool_list)
        # loop guard: not used in a schema request (a grammar is present)
        guard = None
        if constraint is None:
            from .session import LoopGuard
            close_tok = m.tok.convert_tokens_to_ids("</think>")
            finish_id = m.tok.convert_tokens_to_ids("<|im_end|>")
            guard = LoopGuard(close_tok if isinstance(close_tok, int) and close_tok >= 0 else None, finish_id if finish_id in m.eos else sorted(m.eos)[0], lambda: parser.phase == "thinking",
                                   limit=int(pick.get("thinking_limit") or 3000))      # 2026-11-11: 8,000 → 3,000 (on one trip-planning question 8,000 tokens = 81 s of thinking; usual thinking < 1,500)
        detok = m.tok.detokenizer
        detok.reset()
        streaming = bool(body.get("stream", True))
        batch = {"content": "", "thinking": "", "tool_calls": []}

        def emit_events(lst):
            for kind, value in lst:
                if kind == "tool":
                    batch["tool_calls"].append(value)
                    if streaming:
                        row({"tool_calls": [value]})
                else:
                    area = "thinking" if kind == "thinking" else "content"
                    batch[area] += value
                    if streaming:
                        row({area: value})

        try:
            for new in generate(m, o, max_n, sampler, stats, halt=halt, constraint=constraint, no_draft=pick.get("draft") is False, trace=trace, force=guard):
                for b in new:
                    if b not in m.eos:
                        detok.add_token(b)
                chunk = detok.last_segment
                if chunk:
                    emit_events(parser.feed(chunk))
            detok.finalize()
            emit_events(parser.feed(detok.last_segment) + parser.wrap_up())
        except Exception:
            scache.release(o)
            self._release_pending = True        # error during generation: the memory of the discarded session is released when the job returns (second review)
            raise
        stats.generate_secs = max(1e-6, stats.generate_secs - (inter.secs - inter_read))

        if stats.finish == "cancelled":
            scache.release(o)            # a half-finished one is not kept
            self._release_pending = True        # clear_cache on `o` in prune is ineffective while it is alive in this frame; it is released when the job returns
        else:
            o.transient = single         # one-off questions (schema, from tool connectors) go in a small separate pool
        o.busy = False
        self.finish = time.time() + parse_duration(body.get("keep_alive"))
        ns = lambda s: int(s * 1e9)
        latest = {
            "model": name, "created_at": _timestamp(),
            "message": {"role": "assistant", "content": "" if streaming else batch["content"]},
            "done": True, "done_reason": "stop" if stats.finish == "cancelled" else stats.finish,
            "total_duration": ns(time.time() - t_begin), "load_duration": ns(load_time),
            "prompt_eval_count": stats.prompt_tok, "prompt_eval_duration": ns(stats.prompt_secs),
            "eval_count": stats.generated, "eval_duration": ns(stats.generate_secs),
            "engine": {"cache": how, "read": stats.read_tok, "rounds": stats.kind, "draft": stats.draft, "accepted": stats.accept,
                      **_queue_info(background, waited, inter),
                      **({"trace": [([m.tok.decode([t]) for t in tl], k, m.tok.decode([secs])) for tl, k, secs in trace]} if trace else {})},
        }
        if not streaming:
            if batch["thinking"]:
                latest["message"]["thinking"] = batch["thinking"]
            if batch["tool_calls"]:
                latest["message"]["tool_calls"] = batch["tool_calls"]
        outp.put(("final", latest))
        # 0.4.4: budget by loaded families (formerly Qwen fixed at 12 GB) + pruning. AFTER the reply is sent: the Ollama poll (2 s if it does not answer) does not hold up the reply.
        self._apply_budget(protect=o)
        log_fn(f"chat {how} prompt={stats.prompt_tok} read={stats.read_tok} ({stats.prompt_secs:.1f} s) "
               f"generated={stats.generated} ({stats.generated / max(stats.generate_secs, 1e-6):.1f} tok/s) accepted={stats.accept}/{stats.draft} "
               f"tools={len(batch['tool_calls'])} {'schema ' if schema else ''}{stats.finish}" + _queue_log(background, waited, inter)
               + (" GUARD: " + "; ".join(guard.event) if guard is not None and guard.event else "")
               + (f" from disk {begin} tokens {disk_secs:.1f} s (in memory {mem_begin})" if how == "disk" else "") + f" | {scache.digest()}")
        # 2026-11-12: AFTER the reply is sent, if the root snapshots of the system block are not on disk yet, they are written (engine/disk.py).
        # One-off requests (plan step, decision extraction: fixed system block + varying question) are written too; short prefixes (< 1,024) are not.
        if gb is None and stats.finish != "cancelled":
            try:
                from . import disk
                k = disk.save(o, MODEL_IDENT)
                if k:
                    log_fn(f"prefix cache written to disk: {k['position']} tokens, {k['count']} checkpoints, {k['bytes'] / 1e6:.0f} MB, {k['seconds']:.1f} s")
            except Exception as e:
                log_fn(f"prefix cache could not be written to disk ({type(e).__name__}: {e})")

    def chat_gemma(self, body: dict, outp: "queue.Queue", cancel: threading.Event, halt=None, inter: Optional[Interrupt] = None, waited: float = 0.0, background: bool = False):
        """Gemma 4 chat: the same flow as Qwen's; the format (chat_gemma), session and loop (gemma) are family-specific."""
        if halt is None:
            halt, inter = cancel.is_set, Interrupt()
        from . import chat_gemma as G
        from .gemma import DEFAULTS as GV, step_forward, generate
        from .sampling import Sampler
        from .session import LoopGuard, Cancelled, Stats

        t_begin = time.time()
        load_time = self.load_gemma()
        m, scache = self.gemma, self.gemma_session_cache
        pick = body.get("options") or {}
        tool_list = body.get("tools") or None
        schema = body.get("format")
        think = body.get("think", True)
        if schema:
            think = False
        messages, image_list, digests = G.flatten(body.get("messages") or [])
        prompt = G.build_prompt(m.tok, messages, tool_list, think)
        gb = None
        if image_list:
            gb = m.prepare_image(prompt, image_list)
            ids, places = gb["ids"], gb["positions"]
        else:
            ids, places = m.tok.encode(prompt, add_special_tokens=False), []
        if len(ids) >= WINDOW:
            raise ValueError(f"prompt {len(ids)} tokens; window {WINDOW}")

        single = bool(schema) or body.get("cache") is False
        o, how = scache.pick(ids, digests, places, transient=single)
        begin = len(o.tok)
        image = None
        if gb is not None and any(y + n > begin for y, n in zip(gb["positions"], gb["counts"])):
            image = (m.image_embed(gb, digests, begin), gb["mm"])
        # snapshot locations: the end of the system block (if long) and the start of the last turn (the next message of the same chat continues from here).
        # Each snapshot is the sliding window itself (≈ 839 MB) → the 512-token intermediate checkpoints of Qwen do not exist here.
        turn_start = m.tok.convert_tokens_to_ids("<|turn>")
        starts = [i for i, b in enumerate(ids) if b == turn_start]
        place = set()
        if len(starts) >= 2:
            if not single:                    # a one-off request (schema / cache: false) does not continue: a turn-start snapshot would be a wasted copy
                place.add(starts[-1])
            if starts[1] >= 512:
                place.add(starts[1])
                o.root_limit = max(o.root_limit, starts[1])
        if gb is not None and gb["positions"]:
            # end of the last image: when a different question is asked about the same image the image is not re-read
            place.add(gb["positions"][-1] + gb["counts"][-1] + 1)
        # Learned checkpoint (2026-11-11): if this prompt is identical to a session in the cache up to ≥ 512 tokens BEYOND a usable snapshot
        # (e.g. a shared 1,355-token rule block of a batch writer; where the connector rules begin in the client's prompt), a snapshot is taken at the
        # divergence point and becomes a root → the next request with the same prefix forks from there. The counterpart of Ollama's "divergence point seen a second time" checkpoint.
        learned = getattr(scache, "last_common", 0)
        if learned - begin >= (LEARN_SHORT if learned <= 512 else LEARN_MIN) and learned < len(ids) - 16 and not (gb is not None and any(y <= learned < y + n for y, n in zip(gb["positions"], gb["counts"]))):
            place.add(learned)
            o.root_limit = max(o.root_limit, learned)
        stats = Stats(prompt_tok=len(ids), read_tok=len(ids) - begin)
        name = body.get("model") or GEMMA_NAME

        def row(message: dict, **extra):
            d = {"model": name, "created_at": _timestamp(),
                 "message": {"role": "assistant", "content": "", **message}, "done": False}
            d.update(extra)
            outp.put(("line", d))

        t = time.perf_counter()
        inter.session, o.busy = o, True     # while the job runs (even if paused) the pool does not reuse / discard this session in place
        self._budget_before_read("gemma", o, len(ids) - begin)
        try:
            step_forward(m, o, ids, image=image, snapshot_places=place, halt=halt,
                   image_ends=[y + n + 1 for y, n in zip(gb["positions"], gb["counts"])] if gb is not None else ())
        except Cancelled:
            scache.release(o)
            inter.session = None
            self._release_pending = True        # the memory is released when the job function returns (the traceback holds the session here)
            outp.put(("final", None))
            log_fn(f"gemma cancelled (while reading) prompt={len(ids)}")
            return
        except Exception:
            scache.release(o)
            inter.session = None
            self._release_pending = True
            raise
        o.image_list = list(digests)
        stats.prompt_secs = time.perf_counter() - t - inter.secs      # time spent in interleaved jobs does not count toward the prefill time
        inter_read, inter.read_n = inter.secs, inter.n

        sampler = Sampler(
            temperature=float(pick.get("temperature", GV["temperature"])),
            top_k=int(pick.get("top_k", GV["top_k"])),
            top_p=float(pick.get("top_p", GV["top_p"])),
            min_p=float(pick.get("min_p", GV["min_p"])),
            rng_seed=pick.get("seed"),
        )
        max_n = int(pick.get("num_predict", -1) or -1)
        if max_n <= 0:
            max_n = max(16, min(32768, WINDOW - len(ids)))
        constraint = None
        if schema:
            from .constraint import Constraint

            constraint = Constraint(m.tok._tokenizer, {"type": "object"} if schema == "json" else schema)

        trace = [] if pick.get("trace") else None
        parser = G.Parser(bool(think), tool_list, thinking_on=G.starts_thinking(prompt))
        guard = None
        if constraint is None:
            close_tok = m.tok.convert_tokens_to_ids("<channel|>")
            finish_id = m.tok.convert_tokens_to_ids("<turn|>")
            guard = LoopGuard(close_tok if isinstance(close_tok, int) and close_tok >= 0 else None, finish_id if finish_id in m.eos else sorted(m.eos)[0], lambda: parser.phase == "thinking",
                                   limit=int(pick.get("thinking_limit") or 3000))
        detok = m.tok.detokenizer
        detok.reset()
        streaming = bool(body.get("stream", True))
        batch = {"content": "", "thinking": "", "tool_calls": []}

        def emit_events(lst):
            for kind, value in lst:
                if kind == "tool":
                    batch["tool_calls"].append(value)
                    if streaming:
                        row({"tool_calls": [value]})
                else:
                    area = "thinking" if kind == "thinking" else "content"
                    batch[area] += value
                    if streaming:
                        row({area: value})

        try:
            for new in generate(m, o, max_n, sampler, stats, halt=halt, constraint=constraint, no_draft=pick.get("draft") is False, trace=trace, force=guard):
                for b in new:
                    if b not in m.eos:
                        detok.add_token(b)
                chunk = detok.last_segment
                if chunk:
                    emit_events(parser.feed(chunk))
            detok.finalize()
            emit_events(parser.feed(detok.last_segment) + parser.wrap_up())
        except Exception:
            scache.release(o)
            self._release_pending = True        # error during generation: the memory of the discarded session is released when the job returns (second review)
            raise
        stats.generate_secs = max(1e-6, stats.generate_secs - (inter.secs - inter_read))

        if stats.finish == "cancelled":
            scache.release(o)
            self._release_pending = True        # clear_cache on `o` in prune is ineffective while it is alive in this frame; it is released when the job returns
        else:
            o.transient = single
        o.busy = False
        # Session pool budget from free GPU memory: limit − Gemma − (if loaded) the Qwen engine − what Ollama holds − 8 GB working margin; between 3 and 10 GB.
        # Measured (2026-11-11): with Qwen loaded in Ollama (18.6 GB) and a 10 GB pool the total hit the 54 GB limit and reading 7 tokens took 2.7 s.
        # 0.4.4: the budget is now computed in one place for both families (pool_budgets); Qwen's pool also shrinks while Gemma is loaded. Applied after the reply is sent.
        self.gemma_finish = time.time() + parse_duration(body.get("keep_alive"))
        ns = lambda s: int(s * 1e9)
        latest = {
            "model": name, "created_at": _timestamp(),
            "message": {"role": "assistant", "content": "" if streaming else batch["content"]},
            "done": True, "done_reason": "stop" if stats.finish == "cancelled" else stats.finish,
            "total_duration": ns(time.time() - t_begin), "load_duration": ns(load_time),
            "prompt_eval_count": stats.prompt_tok, "prompt_eval_duration": ns(stats.prompt_secs),
            "eval_count": stats.generated, "eval_duration": ns(stats.generate_secs),
            "engine": {"cache": how, "read": stats.read_tok, "rounds": stats.kind, "draft": stats.draft, "accepted": stats.accept,
                      **_queue_info(background, waited, inter),
                      **({"trace": [([m.tok.decode([t]) for t in tl], k, m.tok.decode([secs])) for tl, k, secs in trace]} if trace else {})},
        }
        if not streaming:
            if batch["thinking"]:
                latest["message"]["thinking"] = batch["thinking"]
            if batch["tool_calls"]:
                latest["message"]["tool_calls"] = batch["tool_calls"]
        outp.put(("final", latest))
        self._apply_budget(protect=o)
        if stats.finish != "cancelled":
            o.consolidate()                   # the copy of the snapshots AFTER the reply is sent (it does not count toward the request's time)
        log_fn(f"gemma chat {how} prompt={stats.prompt_tok} read={stats.read_tok} ({stats.prompt_secs:.1f} s) "
               f"generated={stats.generated} ({stats.generated / max(stats.generate_secs, 1e-6):.1f} tok/s) accepted={stats.accept}/{stats.draft} "
               f"tools={len(batch['tool_calls'])} {'schema ' if schema else ''}{stats.finish}" + _queue_log(background, waited, inter)
               + (" GUARD: " + "; ".join(guard.event) if guard is not None and guard.event else "") + f" | {scache.digest()}")


def _queue_info(background: bool, waited: float, inter: Interrupt) -> dict:
    """For the response's `engine` field: the wait time in the queue; for a background job, the number of jobs that interleaved and the pause time."""
    d = {"priority": "background" if background else "front", "wait_s": round(waited, 2)}
    if inter.n:
        d.update({"interleaved": inter.n, "interleaved_while_reading": inter.read_n, "paused_s": round(inter.secs, 2)})
    return d


def _queue_log(background: bool, waited: float, inter: Interrupt) -> str:
    return ((" BACKGROUND" if background else "") + (f" waited={waited:.1f} s" if waited >= 0.5 else "")
            + (f" paused={inter.secs:.1f} s ({inter.n} job(s) interleaved)" if inter.n else ""))


WORKER = Worker()


class RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "FarabiEngine/" + VERSION

    def log_message(self, *a):
        pass

    def _send_json(self, code: int, data: dict):
        b = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _dropped(self) -> bool:
        try:
            r, _, _ = select.select([self.connection], [], [], 0)
            return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except Exception:
            return True

    def do_GET(self):
        file_path = self.path.split("?")[0]
        if file_path == "/api/version":
            return self._send_json(200, {"version": "farabi-engine-" + VERSION})
        if file_path == "/api/tags":
            def size_of(file_path):
                return sum(os.path.getsize(os.path.join(file_path, f)) for f in os.listdir(file_path) if f.endswith(".safetensors")) if os.path.isdir(file_path) else 0
            lst = []
            if size_of(MODEL_PATH):
                lst.append({"name": MODEL_NAME, "model": MODEL_NAME, "size": size_of(MODEL_PATH),
                              "details": {"format": "safetensors", "family": "qwen3_5", "parameter_size": "27.8B", "quantization_level": "nvfp4"}})
            if gemma_installed():
                lst.append({"name": GEMMA_NAME, "model": GEMMA_NAME, "size": size_of(GEMMA_PATH) + size_of(GEMMA_DRAFT),
                              "details": {"format": "safetensors", "family": "gemma4", "parameter_size": "31.7B", "quantization_level": "nvfp4"}})
            return self._send_json(200, {"models": lst})
        if file_path == "/api/ps":
            def record(name, gb, remaining):
                finish = "9999-12-31T00:00:00Z" if remaining == float("inf") else datetime.datetime.fromtimestamp(remaining, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                return {"name": name, "model": name, "size": gb << 30, "size_vram": gb << 30, "context_length": WINDOW, "expires_at": finish}
            def session_cache_gb(scache):       # the session cache also sits in GPU memory (≤ 8 GB for Gemma): so memory guards see the truth
                try:
                    return int(scache.total_memory()) >> 30 if scache is not None else 0
                except Exception:
                    return 0
            lst = []
            if WORKER.engine is not None:
                lst.append(record(MODEL_NAME, SIZE_GB["qwen"], WORKER.finish))
            if WORKER.gemma is not None:
                lst.append(record(GEMMA_NAME, SIZE_GB["gemma"] + session_cache_gb(WORKER.gemma_session_cache), WORKER.gemma_finish))
            return self._send_json(200, {"models": lst})
        if file_path == "/":
            return self._send_json(200, {"status": "Farabi is running", "version": VERSION, "loaded": WORKER.engine is not None,
                                    "gemma_loaded": WORKER.gemma is not None, "gemma": os.path.basename(GEMMA_PATH) if gemma_installed() else None,
                                    "busy": WORKER.busy, "queued": WORKER.job_queue.tally(), "preempting": WORKER.interleaved, "preemptions": WORKER.preempted_count,
                                    "network_lock": NET_LOCK, "model": os.path.basename(MODEL_PATH),
                                    "last_external_request_s": {a: (round(time.time() - t) if t else None) for a, t in WORKER.last_external.items()},
                                    "model_switches": WORKER.model_switches,
                                    "effort": os.environ.get("FARABI_EFFORT", "medium")})
        self._send_json(404, {"error": "unknown endpoint"})

    def do_POST(self):
        file_path = self.path.split("?")[0]
        try:
            body = self._body()
        except Exception as e:
            return self._send_json(400, {"error": f"could not read body: {e}"})
        if file_path == "/api/show" and is_gemma(body.get("model") or body.get("name")):
            return self._send_json(200, {
                "capabilities": ["completion", "vision", "tools", "thinking"],
                "details": {"format": "safetensors", "family": "gemma4", "parameter_size": "31.7B", "quantization_level": "nvfp4"},
                "model_info": {"general.architecture": "gemma4", "gemma4.context_length": WINDOW, "gemma4.block_count": 60},
                "parameters": "temperature 1\ntop_k 64\ntop_p 0.95",
            })
        if file_path == "/api/show":
            return self._send_json(200, {
                "capabilities": ["completion", "vision", "tools", "thinking"],
                "details": {"format": "safetensors", "family": "qwen3_5", "parameter_size": "27.8B", "quantization_level": "nvfp4"},
                "model_info": {"general.architecture": "qwen3_5", "qwen3_5.context_length": WINDOW, "qwen3_5.block_count": 64},
                "parameters": "temperature 1\ntop_k 20\ntop_p 0.95",
            })
        if file_path in ("/api/embed", "/api/embeddings"):
            fed = body.get("input") if file_path == "/api/embed" else body.get("prompt")
            texts = [fed] if isinstance(fed, str) else list(fed or [])
            if not texts:
                return self._send_json(200, {"model": "bge-m3", "embeddings": []} if file_path == "/api/embed" else {"embedding": []})
            result: dict = {}
            finished = threading.Event()
            duration = parse_duration(body.get("keep_alive"), 1800.0)
            t0 = time.time()

            def job_():
                try:
                    result["v"], result["load"], result["tok"] = WORKER.embed(texts, duration)
                except Exception as e:
                    log_fn("embedding error:", traceback.format_exc()[-600:])
                    result["error"] = f"{type(e).__name__}: {e}"
                finally:
                    finished.set()

            WORKER.job_queue.put(job_, FRONT, "embed")        # embedding is a short job (≈ 1 s) and part of the chat flow (memory search): it does not hold back a background job, it interleaves
            if not finished.wait(timeout=600) or "error" in result:
                return self._send_json(500, {"error": result.get("error", "timed out")})
            if file_path == "/api/embeddings":
                return self._send_json(200, {"embedding": result["v"][0]})
            return self._send_json(200, {"model": body.get("model") or "bge-m3", "embeddings": result["v"],
                                    "total_duration": int((time.time() - t0) * 1e9), "load_duration": int(result["load"] * 1e9),
                                    "prompt_eval_count": result["tok"]})
        if file_path == "/api/generate" or (file_path == "/api/chat" and not body.get("messages")):
            # call without a prompt: load the model or (keep_alive 0) unload it
            if body.get("prompt"):
                return self._send_json(400, {"error": "this server generates only via /api/chat"})
            finished = threading.Event()
            duration = parse_duration(body.get("keep_alive"))
            gemma = is_gemma(body.get("model"))

            def job_():
                try:
                    if gemma:
                        if duration == 0:
                            WORKER.unload_gemma("request")
                        else:
                            WORKER.load_gemma()
                            WORKER.gemma_finish = time.time() + duration
                    elif duration == 0:
                        WORKER.unload("request")
                    else:
                        WORKER.do_load()
                        WORKER.finish = time.time() + duration
                finally:
                    finished.set()

            WORKER.job_queue.put(job_, NORMAL)                # load / unload: does not interleave (so it does not unload the model of a paused job), waits for the running job to finish
            finished.wait(timeout=300)
            return self._send_json(200, {"model": GEMMA_NAME if gemma else MODEL_NAME, "done": True, "done_reason": "unload" if duration == 0 else "load"})
        if file_path != "/api/chat":
            return self._send_json(404, {"error": "unknown endpoint"})

        outp: "queue.Queue" = queue.Queue()
        cancel = threading.Event()

        def job_():
            try:
                WORKER.chat(body, outp, cancel)
            except Exception as e:
                log_fn("chat error:", traceback.format_exc()[-1200:])
                outp.put(("error", f"{type(e).__name__}: {e}"))

        WORKER.job_queue.put(job_, parse_priority(body), "gemma" if is_gemma(body.get("model")) else "qwen")
        streaming = bool(body.get("stream", True))
        headers_sent = False
        while True:
            try:
                kind, data = outp.get(timeout=0.25)
            except queue.Empty:
                if self._dropped():
                    cancel.set()
                continue
            try:
                if kind == "error":
                    if not headers_sent:
                        return self._send_json(500, {"error": data})
                    self.wfile.write((json.dumps({"error": data}, ensure_ascii=False) + "\n").encode())
                    return
                if kind == "final" and data is None:
                    return                      # client is gone
                if not streaming:
                    if kind == "final":
                        return self._send_json(200, data)
                    continue
                if not headers_sent:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    headers_sent = True
                self.wfile.write((json.dumps(data, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()
                if kind == "final":
                    return
            except (BrokenPipeError, ConnectionResetError, OSError):
                cancel.set()
                return


class Server(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer tries to resolve the host name here (DNS); avoid pointless waiting under the lock
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("FARABI_PORT", 11438)))
    ap.add_argument("--preload", action="store_true", help="load the model at startup")
    a = ap.parse_args()
    global NET_LOCK
    net_lock_setup()                      # if not locked, the process restarts here inside the sandbox
    NET_LOCK = net_lock_active()
    if not NET_LOCK and os.environ.get("FARABI_NETWORK_LOCK") != "off":
        raise SystemExit("network lock is not active: the process can open outbound connections; server not started")
    WORKER.start()
    if a.preload:
        def job_():
            WORKER.do_load()
            WORKER.finish = time.time() + 600
        WORKER.job_queue.put(job_, NORMAL)
    server = Server(("127.0.0.1", a.port), RequestHandler)
    server.daemon_threads = True
    log_fn(f"Farabi engine {VERSION} listening: 127.0.0.1:{a.port} (pid {os.getpid()}) | network lock: {'ACTIVE' if NET_LOCK else 'INACTIVE'} | model: {os.path.basename(MODEL_PATH)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
