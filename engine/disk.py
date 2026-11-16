"""On-disk copy of the prefix cache (2026-11-12).

Why: every chat of the client app starts with the same long system prompt (≈ 9.3k tokens; reading it takes ≈ 12 s). The engine keeps it in memory,
but when the model went idle and was unloaded, or the engine restarted, the cache went with it -> the "first message after a break" was always expensive.

What is stored (Qwen path, engine/session.py): the root snapshots of the system block —
  the keys/values of the 16 attention layers [0, K) + those of the MTP head [0, K-1) + at chosen checkpoints (about every 2,048 tokens and the last)
  the state of the 48 linear layers and that slot's hidden state. About 1.3 GB at 9.3k tokens.
The file name is the digest of the prefix's tokens; the .json beside it carries each checkpoint's digest -> when the prompt changes (a new memory note, a new decision)
the LONGEST matching checkpoint is loaded and only the changed tail is read again. At most MAX_RECORDS files per model; the least recently used is deleted.
FARABI_PREFIX_CACHE=off disables it; if another path is given, that path is used. Prefixes containing images and short prefixes (< MIN_LEN) are not written.

2026-11-12 (second round, from the test log): (1) the "write on second sighting" rule now works at CHECKPOINT level — even if the end of the system block changes
from request to request (decisions, memory notes, same tool set but a different tail), the longest shared part seen twice is written; if the part missing
from disk is >= FIRST_WRITE (a new tool set) it is written on first sight; (2) if at least (K − WRITE_GAIN) is already on disk, no new file is written; (3) the server uses the disk not only when "there is no match at all in memory" but
also when the in-memory match is SHORTER than the disk's (longest + do_load); (4) MAX_RECORDS 6 -> 16 (a test had pushed the everyday prompt record out).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import List, Optional

import mlx.core as mx
import numpy as np

from .session import Snapshot, Session

_ENV = os.environ.get("FARABI_PREFIX_CACHE", "")
ROOT = _ENV if _ENV not in ("", "off") else os.path.expanduser("~/.farabi/prefix-cache")
FORMAT = 1
MIN_LEN = 1024            # prefixes shorter than this are not written to disk (reading them takes about 1 s anyway)
MAX_PREFIX = 32768     # prefixes longer than this are not written to disk (2026-11-14, 0.4.4): the client's long system prompt is 9-16k; anything longer is a one-off giant message
INTERVAL = 2048           # approximate spacing of the intermediate checkpoints written to disk (each checkpoint ≈ 151 MB)
MAX_RECORDS = int(os.environ.get("FARABI_PREFIX_CACHE_MAX", 16))      # ≈ 1.4-2.4 GB per file -> at most ≈ 35 GB
FIRST_WRITE = 6144          # if the part missing from disk is this long (a new tool set), a second sighting is not awaited and it is written immediately
WRITE_GAIN = 1536       # if the longest match on disk leaves a gap shorter than this, no new file is written (so no file is opened for the shared 1,024-token tool prefix)
_seen: dict = {}     # checkpoint digest -> in how many requests it was seen in this process (written to disk on the second)
_meta: dict = {}        # sidecar file name -> content (sidecar files do not change once written)


def enabled() -> bool:
    return _ENV != "off"


def _digest(ids) -> str:
    return hashlib.sha1(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()


def _chosen(roots: List[Snapshot]) -> List[Snapshot]:
    """The last checkpoint + the two before it + the first checkpoint past each multiple of INTERVAL.

    The two extra checkpoints at the end: the client app puts blocks that change over time in the prompt (decisions, memory notes) at the end; when a new note is added the divergence
    point is a few hundred tokens before the end -> the nearest checkpoint should be at most 512 tokens behind (otherwise up to 2,048 tokens would be re-read)."""
    pick, tgt = [], INTERVAL
    for a in roots[:-1]:
        if a.location >= tgt:
            pick.append(a)
            tgt = (a.location // INTERVAL + 1) * INTERVAL
    for a in roots[-3:-1]:
        if a not in pick:
            pick.append(a)
    return sorted(pick, key=lambda a: a.location) + [roots[-1]]


def _prune(folder: str):
    records = sorted((f for f in os.listdir(folder) if f.endswith(".json")), key=lambda f: os.path.getmtime(os.path.join(folder, f)))
    for f in records[:-MAX_RECORDS]:
        for file_path in (os.path.join(folder, f), os.path.join(folder, f[:-5] + ".safetensors")):
            try:
                os.remove(file_path)
            except OSError:
                pass


def save(o: Session, ident: str) -> Optional[dict]:
    """Writes the session's root snapshots to disk; returns the digest if it wrote, None if there was no need."""
    if not enabled() or o.root_limit < MIN_LEN:
        return None
    roots = sorted((a for a in o.snapshots if a.location <= o.root_limit and a.n_images == 0 and a.shift == 0), key=lambda a: a.location)
    if not roots or roots[-1].location < MIN_LEN:
        return None
    K = roots[-1].location
    if K > MAX_PREFIX:
        return None          # 0.4.4: a single giant message with no system message must not count as a "prefix" and be written to disk (100k tokens ≈ a 7 GB file)
    if any(c.keys is None or c.offset < K for c in o.kv) or (o.cm and (o.cm[0].keys is None or o.cm[0].offset < K - 1)):
        return None
    folder = os.path.join(ROOT, ident)
    os.makedirs(folder, exist_ok=True)
    # Counting "seen a second time" at checkpoint level (2026-11-12): the prefix of each root checkpoint is counted separately -> even if the tail of the system block changes
    # from request to request (selected decisions, a new memory note), the shared part is written. One-off prefixes still do not go to disk.
    h, digests, previous = hashlib.sha1(), {}, 0
    arr = np.asarray(o.tok[:K], dtype=np.int32)
    for a in roots:
        h.update(arr[previous:a.location].tobytes()); previous = a.location
        digests[a.location] = h.copy().hexdigest()
    for oz in digests.values():
        _seen[oz] = _seen.get(oz, 0) + 1
    if len(_seen) > 8192:
        for k in list(_seen)[:4096]:
            del _seen[k]
    # What is written: (a) if the part missing from disk is >= FIRST_WRITE (a new tool set: reading from scratch takes 10-20 s), the WHOLE prefix, on first sight;
    # (b) otherwise up to the longest checkpoint seen twice, provided it is at least WRITE_GAIN longer than what is on disk. If neither, only the file in use is refreshed.
    found, file, _m = longest(o.tok[:K], ident, o, inclusive=True)
    if K - found < FIRST_WRITE:
        twice = [a.location for a in roots if a.location >= MIN_LEN and _seen.get(digests[a.location], 0) >= 2]
        K2 = twice[-1] if twice else 0
        if not K2 or K2 - longest(o.tok[:K2], ident, o, inclusive=True)[0] < WRITE_GAIN:
            if file:
                try:
                    os.utime(os.path.join(folder, file))        # used: let it be last when pruning
                except OSError:
                    pass
            return None
        K = K2
    roots = sorted((a for a in o.snapshots if a.location <= K and a.n_images == 0 and a.shift == 0), key=lambda a: a.location)
    name = digests[K]
    file_path, meta_path = os.path.join(folder, name + ".safetensors"), os.path.join(folder, name + ".json")
    t = time.time()
    chosen = _chosen(roots)
    if any(ev is None or dl is None for a in chosen for ev, dl in a.gdn) or any(a.hidden is None for a in chosen):
        return None
    v = {"tok": mx.array(np.asarray(o.tok[:K], dtype=np.int32))}
    for i, c in enumerate(o.kv):
        v[f"kv.{i}.k"], v[f"kv.{i}.v"] = c.keys[..., :K, :], c.values[..., :K, :]
    if o.cm:
        v["cm.k"], v["cm.v"] = o.cm[0].keys[..., : K - 1, :], o.cm[0].values[..., : K - 1, :]
    for a in chosen:
        v[f"a.{a.location}.hidden"] = a.hidden
        for j, (ev, dl) in enumerate(a.gdn):
            v[f"a.{a.location}.g.{j}.ev"], v[f"a.{a.location}.g.{j}.dl"] = ev, dl
    transient = os.path.join(folder, name + ".writing.safetensors")
    mx.save_safetensors(transient, v)
    os.replace(transient, file_path)
    meta = {"format": FORMAT, "position": K, "checkpoints": [a.location for a in chosen],
            "digest": {str(a.location): digests[a.location] for a in chosen},
            "kv": len(o.kv), "gdn": len(o.gdn), "cm": bool(o.cm), "bytes": os.path.getsize(file_path), "written": time.time()}
    with open(meta_path + ".writing", "w") as f:
        json.dump(meta, f)
    os.replace(meta_path + ".writing", meta_path)
    _prune(folder)
    return {"position": K, "count": len(chosen), "bytes": meta["bytes"], "seconds": time.time() - t}


def longest(ids: List[int], ident: str, o: Session, inclusive: bool = False):
    """The longest prefix checkpoint of ids found on disk: (position, sidecar file name, sidecar content); if none, (0, None, None). Loads nothing.

    inclusive=False: the checkpoint must be SHORTER than ids (at least one token must remain to be read after restoring). inclusive=True: the full length also counts (save)."""
    if not enabled():
        return 0, None, None
    folder = os.path.join(ROOT, ident)
    if not os.path.isdir(folder):
        return 0, None, None
    arr, memo = np.asarray(ids, dtype=np.int32), {}

    def oz(k):
        if k not in memo:
            memo[k] = hashlib.sha1(arr[:k].tobytes()).hexdigest()
        return memo[k]
    best = (0, None, None)
    for f in os.listdir(folder):
        if not f.endswith(".json"):
            continue
        meta = _meta.get(f)
        if meta is None:
            try:
                with open(os.path.join(folder, f)) as hd:
                    meta = json.load(hd)
            except (OSError, ValueError):
                continue
            _meta[f] = meta
        if meta.get("format") != FORMAT or meta.get("kv") != len(o.kv) or meta.get("gdn") != len(o.gdn) or bool(meta.get("cm")) != bool(o.cm):
            continue
        for k in sorted(meta.get("checkpoints") or [], reverse=True):
            if k <= best[0]:
                break
            if (k < len(ids) or (inclusive and k == len(ids))) and oz(k) == (meta.get("digest") or {}).get(str(k)):
                if os.path.exists(os.path.join(folder, f[:-5] + ".safetensors")):
                    best = (k, f, meta)
                else:
                    _meta.pop(f, None)
                break
    return best


def do_load(o: Session, ids: List[int], ident: str, K: int, f: str, meta: dict) -> int:
    """Loads the checkpoint found by longest into an EMPTY session. Returns: the loaded position."""
    folder = os.path.join(ROOT, ident)
    v = mx.load(os.path.join(folder, f[:-5] + ".safetensors"))
    for i, c in enumerate(o.kv):
        c.keys, c.values, c.offset = v[f"kv.{i}.k"][..., :K, :], v[f"kv.{i}.v"][..., :K, :], K
    if o.cm:
        o.cm[0].keys, o.cm[0].values, o.cm[0].offset = v["cm.k"][..., : K - 1, :], v["cm.v"][..., : K - 1, :], K - 1
    o.snapshots = [Snapshot(k, [(v[f"a.{k}.g.{j}.ev"], v[f"a.{k}.g.{j}.dl"]) for j in range(len(o.gdn))], v[f"a.{k}.hidden"], 0, 0)
                  for k in sorted(meta["checkpoints"]) if k <= K]
    latest = o.snapshots[-1]
    for c, (ev, dl) in zip(o.gdn, latest.gdn):
        c[0], c[1] = ev, dl
    o.tok = list(ids[:K])
    o.hidden, o.shift, o.image_list, o.root_limit = latest.hidden, 0, [], K
    mx.eval(o.states(), [x for a in o.snapshots for pair in a.gdn for x in pair], [a.hidden for a in o.snapshots])
    try:
        os.utime(os.path.join(folder, f))
    except OSError:
        pass
    return K


def restore(o: Session, ids: List[int], ident: str) -> int:
    """Loads, into an EMPTY session, the longest prefix checkpoint of ids found on disk. Returns: the loaded position (0 = no match)."""
    if not enabled() or o.tok:
        return 0
    K, f, meta = longest(ids, ident, o)
    return do_load(o, ids, ident, K, f, meta) if K else 0
