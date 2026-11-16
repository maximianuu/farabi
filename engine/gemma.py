"""Gemma 4 family - core, session and generation loop (the counterpart of Qwen's engine/core.py + engine/session.py).

Differences from Qwen:
- Of the main model's 60 layers, 50 use sliding-window attention (1,024) and 10 use full attention. A sliding-window cache cannot be
  rewound -> the snapshot is the window itself (50 layers × 1,024 entries ≈ 839 MB); the full-attention cache
  is not copied, only its length is pulled back.
- Drafts come from a separate model (Google's official Gemma4Assistant: 4 layers, it shares the main model's last layers'
  keys/values and has no cache of its own).
- Verification is done on SHADOW copies of the cache: a multi-token append binds the copy to new arrays, the real cache
  stays unchanged; after acceptance only the accepted tokens are fed in. There is no rollback (transaction log).

Written against mlx-vlm 0.7.12.
"""

from __future__ import annotations

import copy
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import mlx.core as mx
import numpy as np
from mlx_vlm.models.cache import RotatingKVCache

from .depth import Depth
from .sampling import Sampler
from .session import Cancelled, Stats, SessionCache, _common_prefix

FINISH = ("<eos>", "<turn|>", "<|tool_response>")
# Soft token budget per image (Gemma 4: 280 / 560 / 1,120). Ollama picks it by the image's pixel area - measured on 29 frames on 2026-11-11
# (from the prompt token count): <= 574,690 px -> ~264 tokens · 600,000-1,166,400 px -> ~530 · >= 1,306,200 px -> ~1,090. The thresholds lie somewhere in these gaps;
# 768² and 2×768² were chosen. The library's default is always 280 (small text in a large image gets lost). Pin it with FARABI_GEMMA_IMAGE_BUDGET=280|560|1120.
IMAGE_THRESH = ((589_824, 280), (1_179_648, 560))
IMAGE_MAX = 1120


def image_budget(width: int, height: int) -> int:
    import os
    fixed = os.environ.get("FARABI_GEMMA_IMAGE_BUDGET", "")
    if fixed.isdigit():
        return int(fixed)
    area = width * height
    return next((b for thresh, b in IMAGE_THRESH if area <= thresh), IMAGE_MAX)
DEFAULTS = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0}     # Google's generation_config; same as in Ollama


def _argmax(l):
    return mx.argmax(l, axis=-1)


# ---------------------------------------------------------------- fast attention (full-attention layers)
# In the full-attention layers, 32 query heads share 4 key/value heads (8 each). The library path (mx.fast.scaled_dot_product_attention) is slow in this
# case: measured (2026-11-11, single query): 0.63 ms at 5,000 keys vs 0.29 ms grouped; 1.09 vs 0.52 ms at 11,000; 3-8 ms per round over 10 layers.
# Grouped computation: queries are reshaped to (B, 4, 8·L, D) and keys/values are not replicated; the result is mathematically identical (softmax is taken in f32).
# Only kicks in at generation/verification length (<= 16 queries) and when the sharing is >= 4; no gain in the sliding layers (sharing of 2) -> library path.
_ATTN_INSTALLED = False


def _install_fast_attn():
    global _ATTN_INSTALLED
    if _ATTN_INSTALLED:
        return
    from mlx_vlm.models.gemma4 import language as L

    original = L.scaled_dot_product_attention

    def attn(queries, keys, values, cache, scale, mask, sinks=None):
        B, H, n, D = queries.shape
        G = keys.shape[1]
        if (sinks is not None or n > 16 or G == 0 or H % G or H // G < 4 or hasattr(cache, "bits") or keys.dtype not in (mx.bfloat16, mx.float16, mx.float32)
                or not (mask is None or isinstance(mask, str) or (isinstance(mask, mx.array) and mask.dtype == mx.bool_))):
            return original(queries, keys, values, cache, scale, mask, sinks) if sinks is not None else original(queries, keys, values, cache, scale, mask)
        R, S = H // G, keys.shape[2]
        q = queries.reshape(B, G, R * n, D)
        s = (q if scale == 1.0 else q * scale) @ keys.swapaxes(-1, -2)                 # (B, G, R·n, S)
        allowed = None
        if isinstance(mask, str):
            if n > 1:                                                                    # causal: the i-th query cannot see keys after itself among the last n keys
                allowed = mx.arange(S)[None, :] <= (mx.arange(n)[:, None] + (S - n))
        elif mask is not None:
            allowed = mask.reshape(-1, n, S)[0] if mask.size == n * S else None
            if allowed is None:
                return original(queries, keys, values, cache, scale, mask)
        if allowed is not None:
            s = mx.where(mx.tile(allowed, (R, 1)), s, mx.array(-mx.inf, dtype=s.dtype))
        w = mx.softmax(s.astype(mx.float32), axis=-1).astype(queries.dtype)
        return (w @ values).reshape(B, H, n, D)

    L.scaled_dot_product_attention = attn
    _ATTN_INSTALLED = True


# The tokenizer + image processor STAY IN THE PROCESS even after the model is unloaded (2026-11-12, engine 0.4.1). Measured: of a warm
# load's 2.4 s, ~2.0 s was not the weights but the 30 MB tokenizer.json being read and parsed twice (the processor's tokenizer +
# the chat tokenizer); the weights take ~0.5 s. These are CPU objects: they hold no GPU memory and carry no model-bound state (the detokenizer hands out a
# copy on every call) -> reused as they are on reload. If the config files in the model folder change (signature), they are rebuilt.
# The cost is RAM: the two objects together hold ~1.1 GB -> not kept forever: GEMMA_PERSIST_SECS (30 min) after Gemma is unloaded, the
# worker's idle check releases these too (Worker._release_persistent).
# FARABI_GEMMA_PERSIST=0 returns to the old path (the library's `load` on every load).
_PERSISTENT: dict = {}        # model path -> (signature, processor, chat tokenizer)


def _config_signature(model_path: str) -> tuple:
    import os
    sig = []
    for name in sorted(os.listdir(model_path)):
        if not name.endswith(".safetensors"):
            try:
                st = os.stat(os.path.join(model_path, name))
                sig.append((name, st.st_mtime_ns, st.st_size))
            except OSError:
                pass
    return tuple(sig)


def _model_and_processor(model_path: str):
    """The SAME objects as `mlx_vlm.load(model_path)` + `mlx_lm.tokenizer_utils.load`; the processor and tokenizer are built once per process.
    Returns: (model, processor, tokenizer, whether it came from storage)."""
    import os
    from mlx_lm.tokenizer_utils import load as tok_loader

    if os.environ.get("FARABI_GEMMA_PERSIST", "1") == "0":
        from mlx_vlm import load

        model, processor = load(model_path)
        return model, processor, tok_loader(Path(model_path)), False
    from mlx_vlm.utils import get_model_path, load_image_processor, load_model, load_processor

    file_path = get_model_path(model_path)
    model = load_model(file_path, False, strict=True)
    signature = _config_signature(model_path)
    k = _PERSISTENT.get(model_path)
    stored = k is not None and k[0] == signature
    if not stored:
        img_processor = load_image_processor(file_path)
        processor = load_processor(file_path, True, eos_token_ids=getattr(model.config, "eos_token_id", None))
        if img_processor is not None:
            processor.image_processor = img_processor
        k = (signature, processor, tok_loader(Path(model_path)))
        _PERSISTENT.clear()                                  # there is a single Gemma model: do not leave objects of the old path behind
        _PERSISTENT[model_path] = k
    if hasattr(model, "attach_processor"):
        model.attach_processor(k[1])
    return model, k[1], k[2], stored


class GemmaEngine:
    family = "gemma"

    def __init__(self, model_path: str, draft_path: str):
        from mlx_vlm.speculative.drafters import load_drafter

        import os as _os
        if _os.environ.get("FARABI_GEMMA_FAST_ATTENTION", "1") != "0":
            _install_fast_attn()
        self.model, self.processor, self.tok, self.stored_processor = _model_and_processor(model_path)
        self.lm = self.model.language_model
        self.body = self.lm.model
        self._apply_norm_eps(model_path)
        self.draft, _ = load_drafter(draft_path)
        self.draft.reset(self.model)                    # binds the draft model to the main model's embedding table
        self.eos = {i for i in (self.tok.convert_tokens_to_ids(x) for x in FINISH) if isinstance(i, int) and i >= 0}
        self.window = int(self.lm.config.sliding_window)
        # depth state learned across requests; the acceptance rate and round cost differ between greedy and sampled decoding
        # (measured: sampled acceptance 25-40%, greedy 50-80%) -> separate controllers, so that they do not corrupt each other's statistics
        self.depth = Depth(deepest=15)
        self.depth_greedy = Depth(deepest=15)
        # MLX's fused attention kernel (head dim 64/80/128 -> vision tower): if the FIRST call in the process is masked, the process stays ~2.7x slower for its whole life
        # (measured 2026-11-11, in two separate processes: 37.7 vs 14.1 ms for 9,900 patches; masks are applied correctly in both cases). The library's vision path
        # always calls with a mask; we use the maskless path (_tower), and so as not to leave it to chance we make the first call here, maskless.
        _q = mx.zeros((1, 16, 8, 80), dtype=mx.bfloat16)
        mx.eval(mx.fast.scaled_dot_product_attention(_q, _q, _q, scale=1.0, mask=None))
        self.vision = self                                # the vision tower lives inside the model file (no separate load)
        self.image_ident = int(self.model.config.image_token_id)
        self._feature: "OrderedDict[str, mx.array]" = OrderedDict()     # image digest -> tower output (1, n, H); ~3 MB per image

    def _apply_norm_eps(self, model_path: str):
        """In our own conversion (tools/gemma_convert3.py) the projections are stored scaled up by 2^k; the RMS norm after them absorbs the scale. To keep the function EXACTLY the same,
        those norms' eps is also scaled up by 2^(2k) (the exponents are in config.json -> farabi_norm_eps_us). If the field is missing (stock conversion), nothing is touched."""
        import json, os
        try:
            exps = json.load(open(os.path.join(model_path, "config.json"))).get("farabi_norm_eps_exp") or {}
        except Exception:
            exps = {}
        for file_path, us in exps.items():
            m = self.body
            for p in file_path.split("."):
                m = m[int(p)] if p.isdigit() else getattr(m, p)
            m.eps = float(m.eps) * (2.0 ** int(us))
        self.norm_eps_count = len(exps)
        # If the vocabulary table was scaled up by 2^k and compressed (tools/gemma_embed.py): the embedding scale is corrected on input and the logit by 2^-k on output.
        # The draft model reads the embedding scale when it is bound -> this must happen BEFORE draft.reset.
        try:
            k = json.load(open(os.path.join(model_path, "config.json"))).get("farabi_embed_shift")
        except Exception:
            k = None
        self._head_scale = (2.0 ** -int(k)) if k else None
        if self._head_scale:
            self.body.embed_scale = float(self.body.embed_scale) * self._head_scale

    def session_cache(self):
        return self.lm.make_cache()

    def hidden(self, fed, cache, embedding=None):
        """Output of the last layer, BEFORE the final norm (the logits and the draft's input are derived from it)."""
        if embedding is None:
            return self.body(fed, cache=cache, skip_final_norm=True)
        return self.body(fed, inputs_embeds=embedding, cache=cache, skip_final_norm=True)

    def logit(self, hidden):
        if self._head_scale is None:
            return self.lm.speculative_logits_from_hidden(hidden)
        from mlx_vlm.models.gemma4.language import logit_softcap

        lg = self.body.embed_tokens.as_linear(self.body.norm(hidden)) * self._head_scale
        limit = self.lm.final_logit_softcapping
        return logit_softcap(limit, lg) if limit is not None else lg

    # -- image
    def prepare_image(self, prompt: str, image_list: list) -> dict:
        """Prompt with images -> tokens + the soft-token range of each image + pixel arrays.

        The template writes one `<|image|>` per image; the processor expands it into the sequence `<|image>` + n × `<|image|>` + `<image|>`
        (n <= 280 / 560 / 1,120 depending on the image size: image_budget). Tokenization uses the same tokenizer as chat.
        """
        ip, isl = self.processor.image_processor, self.processor
        pixels, counts = [], []
        for r in image_list:
            data, num = ip([r], max_soft_tokens=image_budget(*r.size))
            pv = data["pixel_values"]
            pixels.append(mx.array(pv if hasattr(pv, "shape") else pv[0][None]))
            counts.append(int(num[0]))
        if prompt.count(isl.image_token) != len(image_list):
            raise ValueError(f"the prompt has {prompt.count(isl.image_token)} image slots, {len(image_list)} images arrived")
        order = iter(counts)
        txt = re.sub(re.escape(isl.image_token), lambda _: f"{isl.boi_token}{isl.image_token * next(order)}{isl.eoi_token}", prompt)
        ids = self.tok.encode(txt, add_special_tokens=False)
        mm = np.asarray(ids) == self.image_ident
        starts = np.flatnonzero(mm & ~np.concatenate([[False], mm[:-1]])).tolist()
        if len(starts) != len(image_list) or any(int(mm[b : b + n].sum()) != n for b, n in zip(starts, counts)):
            raise ValueError("image tokens are not in the expected layout")
        return {"ids": ids, "positions": starts, "counts": counts, "pixels": pixels, "mm": mm}

    def image_embed(self, gb: dict, digests: List[str], begin: int = 0) -> mx.array:
        """Embeddings of the whole prompt (1, n, H); the tower output is placed in the positions of the images after `begin` (cached by digest)."""
        e = self.body.embed_tokens(mx.array(gb["ids"])[None]) * self.body.embed_scale
        for i, (place, n) in enumerate(zip(gb["positions"], gb["counts"])):
            if place + n <= begin:
                continue
            f = self._feature.get(digests[i])
            if f is None:
                f = self._tower(gb["pixels"][i])
                mx.eval(f)
                self._feature[digests[i]] = f
                while len(self._feature) > 32:
                    self._feature.popitem(last=False)
            else:
                self._feature.move_to_end(digests[i])
            if f.shape[1] != n:
                raise ValueError(f"image {i}: the tower produced {f.shape[1]} tokens, the prompt has {n} slots")
            e[:, place : place + n, :] = f.astype(e.dtype)
        return e

    def _tower(self, pv: mx.array) -> mx.array:
        """Plain path for the vision tower, single image and no padding -> (1, n, H) embedding. Gives the same result as the library's `encode_image` (measured: relative
        difference 0) but is ~2x faster on a large image: (1) with no padding, no empty L×L attention mask is passed; (2) position embeddings are taken by direct
        selection instead of an 800 MB one-to-one array; (3) the 3×3 pooling is done by reshaping and averaging instead of `einsum` with an L×n one-to-one weight array
        (patches are laid out row by row; the processor makes the dimensions multiples of 48). If the conditions do not hold, it falls back to the library's path."""
        vt = self.model.vision_tower
        if pv.ndim != 4 or pv.shape[0] != 1:
            return self.model.encode_image(pv)
        _, _, Hp, Wp = pv.shape
        p_, k_ = vt.patch_size, vt.pooling_kernel_size
        pH, pW = Hp // p_, Wp // p_
        if Hp % p_ or Wp % p_ or pH % k_ or pW % k_:
            return self.model.encode_image(pv)
        location, _, _ = vt._patch_positions_single(Hp, Wp, max_patches=pH * pW)
        pos = mx.array(location[None])                                   # (1, L, 2): (x, y), row by row
        pe = vt.patch_embedder
        x = pe._patchify(pv)
        table = pe.position_embedding_table
        x = x + (table[0][pos[..., 0]] + table[1][pos[..., 1]]).astype(x.dtype)
        h = x
        for layer in vt.encoder.layers:              # evaluate layer by layer: each layer's attention produces an intermediate array of ~3 GB; accumulating them in one graph slows things down
            h = layer(h, pos, None)
            mx.eval(h)
        d = h.shape[-1]
        pool = h.reshape(1, pH // k_, k_, pW // k_, k_, d).astype(mx.float32).mean(axis=(2, 4)).reshape(1, -1, d).astype(h.dtype) * vt.pooler.root_hidden_size
        if vt.config.standardize:
            pool = (pool - vt.std_bias) * vt.std_scale
        return self.model.embed_vision(pool)

    def hidden_image(self, embedding, mm: np.ndarray, cache):
        """Reads a chunk that contains image tokens: the tokens of each image attend to each other BIDIRECTIONALLY
        (the model's setting `use_bidirectional_attention: vision`). The library builds this mask only in a single cacheless pass;
        here it is also built when a cache prefix exists: causal (+ window in sliding layers) mask + free attention within the same image block."""
        N = int(mm.shape[0])
        block = np.where(mm, np.cumsum(mm & ~np.concatenate([[False], mm[:-1]])) - 1, -1)
        same = (block[:, None] == block[None, :]) & (block[:, None] != -1)

        def attn_mask(previous: int, window: Optional[int]):
            r, l = np.arange(previous + N)[None], np.arange(previous, previous + N)[:, None]
            mk = l >= r
            if window:
                mk &= l < r + window
            mk[:, previous:] |= same
            return mx.array(mk)[None, None]

        full_attn = next(c for c in cache if not isinstance(c, RotatingKVCache))
        sliding = next(c for c in cache if isinstance(c, RotatingKVCache))
        m_full_attn, m_sliding = attn_mask(full_attn.offset, None), attn_mask(min(self.window - 1, sliding.offset), self.window)
        body = self.body
        body._make_masks = lambda h, c, mm_ids=None: [m_full_attn if k.layer_type == "full_attention" else m_sliding for k in body.layers]
        try:
            return body(None, inputs_embeds=embedding, cache=cache, skip_final_norm=True)
        finally:
            del body._make_masks

    def draft_hidden(self, hidden):
        return self.lm.speculative_draft_hidden(hidden)


# ---------------------------------------------------------------- session


@dataclass
class Snapshot:
    location: int            # number of tokens in the cache
    window: list         # per sliding layer (keys, values): the last <= window entries in time order
    hidden: mx.array       # hidden state of the position-1 slot
    n_images: int
    shift: int = 0


class Session:
    def __init__(self, m: GemmaEngine):
        self.ct = m.session_cache()
        self.sliding = [c for c in self.ct if isinstance(c, RotatingKVCache)]
        self.full_attn = [c for c in self.ct if not isinstance(c, RotatingKVCache)]
        self.cm = None
        self.tok: List[int] = []
        self.hidden: Optional[mx.array] = None
        self.shift = 0
        self.image_list: List[str] = []
        self.snapshots: List[Snapshot] = []
        self.latest = time.time()
        self.transient = False
        self.root_limit = 0
        self.busy = False                # a job is reading/generating in this session (possibly paused): the pool does not reuse it in place and does not discard it (2026-11-12)

    def states(self):
        return [x for c in self.ct if c.keys is not None for x in (c.keys, c.values)]

    def _window(self):
        p = []
        for c in self.sliding:
            k, v = c._temporal_order(c.keys), c._temporal_order(c.values)
            n = min(c.offset, k.shape[2], c.max_size)
            p.append((k[..., k.shape[2] - n:, :], v[..., v.shape[2] - n:, :]))
        return p

    def take_snapshot(self):
        self.snapshots = [a for a in self.snapshots if a.location < len(self.tok)]
        # each snapshot is ~839 MB: root (end of the system block) + the newest two (including this one); forks share the arrays
        root = [a for a in self.snapshots if a.location <= self.root_limit]
        beyond = [a for a in self.snapshots if a.location > self.root_limit]
        self.snapshots = root + beyond[-1:]
        p = self._window()
        mx.eval([x for kv in p for x in kv])
        self.snapshots.append(Snapshot(len(self.tok), p, self.hidden, len(self.image_list)))

    def snapshot_slice(self, location: int, chunk_end: int, hidden: mx.array, n_images: int):
        """AFTER a multi-token chunk has been read, the snapshot for a `location` inside the chunk (no extra forward pass needed).

        A multi-token append lays the sliding cache out in time order; the trailing (chunk_end - location) entries are the ones after the location. The arrays are
        kept as slices, and copying is deferred to `consolidate()` (after the response has gone out). A slice is bound to the cache's array at that moment; even if
        the cache is later written in place, the slice sees the old contents (MLX arrays are immutable nodes).
        """
        self.snapshots = [a for a in self.snapshots if a.location < location]
        root = [a for a in self.snapshots if a.location <= self.root_limit]
        beyond = [a for a in self.snapshots if a.location > self.root_limit]
        self.snapshots = root + beyond[-1:]
        excess = chunk_end - location
        p = []
        for c in self.sliding:
            latest = c.keys.shape[2] - excess
            begin = max(0, latest - min(location, c.max_size))
            p.append((c.keys[..., begin:latest, :], c.values[..., begin:latest, :]))
        self.snapshots.append(Snapshot(location, p, hidden, n_images))
        self._unconsolidated = True

    def consolidate(self):
        """Copies the snapshots kept as slices into their own arrays (releases the large parent arrays)."""
        if getattr(self, "_unconsolidated", False):
            mx.eval([x for a in self.snapshots for kv in a.window for x in kv])
            self._unconsolidated = False

    def _load_window(self, a: Snapshot):
        for c, (k, v) in zip(self.sliding, a.window):
            # new array objects: so that in-place writes to the cache do not touch the snapshot's arrays
            c.keys, c.values = k[..., : k.shape[2], :], v[..., : v.shape[2], :]
            c.offset, c._idx = a.location, k.shape[2]

    def revert(self, a: Snapshot):
        """Rolls the session back to where the snapshot was taken (in place)."""
        for c in self.full_attn:
            c.offset = a.location                 # entries stay; the next append overwrites them
        self._load_window(a)
        self.tok = self.tok[: a.location]
        self.hidden = a.hidden
        self.image_list = self.image_list[: a.n_images]
        self.snapshots = [x for x in self.snapshots if x.location <= a.location]

    def fork(self, m: GemmaEngine, a: Snapshot) -> "Session":
        """A new, independent session from the snapshot (the source is left untouched)."""
        y = Session(m)
        for tgt, source in zip(y.full_attn, self.full_attn):
            tgt.keys = source.keys[..., : a.location, :]
            tgt.values = source.values[..., : a.location, :]
            tgt.offset = a.location
        y._load_window(a)
        y.tok = self.tok[: a.location]
        y.hidden = a.hidden
        y.image_list = self.image_list[: a.n_images]
        y.snapshots = [Snapshot(x.location, x.window, x.hidden, x.n_images) for x in self.snapshots if x.location <= a.location]
        if not any(x.location == a.location for x in y.snapshots):
            y.snapshots.append(Snapshot(a.location, a.window, a.hidden, a.n_images))
        y.root_limit = min(self.root_limit, a.location)
        mx.eval(y.states())
        return y

    def memory(self, seen: Optional[set] = None) -> int:
        """Approximate bytes: live caches + snapshots (if `seen` is given, shared arrays are counted once)."""
        b = 0
        for c in self.full_attn:
            if c.keys is not None:
                b += int((c.keys.nbytes + c.values.nbytes) * min(1.0, c.offset / max(c.keys.shape[2], 1)))
        for c in self.sliding:
            if c.keys is not None:
                b += c.keys.nbytes + c.values.nbytes
        for a in self.snapshots:
            for k, v in a.window:
                for d in (k, v):
                    if seen is None:
                        b += d.nbytes
                    elif id(d) not in seen:
                        seen.add(id(d))
                        b += d.nbytes
        return b


class GemmaSessionCache(SessionCache):
    """Session pool; the selection rule is the same as Qwen's (longest common prefix / snapshot)."""

    def __init__(self, m: GemmaEngine, max_sessions: int = 6, max_bytes: int = 10 << 30, min_len: int = 24):
        # min_len 24 (64 on Qwen): on Gemma a snapshot at a small position is a few MB (the sliding window is only as large as the position); even the
        # 40-70-token shared head of template prompts saves 55-90 ms of reading. An accidental 4-5-token overlap such as `<bos><|turn>user\n` stays below the threshold.
        super().__init__(m, max_sessions, max_bytes, min_len)
        self.max_transient = 2
        self.last_common = 0                 # longest common prefix with any session at the last pick() call (for the learned checkpoint)

    def prune(self, protect: Optional[Session] = None):
        """When the budget is exceeded, old chats forked from the SAME system prompt are evicted first; different system prompts (tool sets) are kept.

        Measurement (2026-11-11, with the client app's 11k-token prompts): one chat is ~3.4 GB (root snapshot 0.84 + per-turn 0.84 + live window 0.84 +
        full attention 0.93); each new chat from the same root costs +2.6 GB. Rebuilding the root takes 15 s, re-reading a chat from the root takes 1-5 s -> the root is more valuable.
        """
        transient = sorted((x for x in self.sessions if x.transient), key=lambda x: x.latest)
        for x in transient[:-self.max_transient]:
            if x is not protect and not getattr(x, "busy", False):
                self.sessions.remove(x)

        def root(o):
            a = next((a for a in o.snapshots if a.location <= o.root_limit and a.window), None)
            return id(a.window[0][0]) if a is not None else None

        def carrying():
            persistent = sum(1 for x in self.sessions if not x.transient)
            return persistent > self.max_sessions or self.total_memory() > self.max_bytes

        while carrying():
            candidates = [x for x in self.sessions if x is not protect and not getattr(x, "busy", False)]      # the session of a paused job is not discarded
            if not candidates:
                break
            tally = Counter(k for k in (root(x) for x in self.sessions) if k is not None)
            with_siblings = [x for x in candidates if tally.get(root(x), 0) > 1]
            pool = with_siblings or [x for x in candidates if not x.transient] or candidates
            self.sessions.remove(min(pool, key=lambda x: x.latest))
        mx.clear_cache()

    def pick(self, ids: List[int], digests: List[str], image_places: List[int], transient: bool = False):
        best = (0, None, None, None)
        self.last_common = 0
        for o in self.sessions:
            l = _common_prefix(o.tok, ids)
            for i, place in enumerate(image_places):
                if place >= l:
                    break
                if i >= len(o.image_list) or o.image_list[i] != digests[i]:
                    l = min(l, place)
                    break
            self.last_common = max(self.last_common, l)
            busy = getattr(o, "busy", False)       # the paused job's session: forked only from its snapshot (see session.SessionCache.pick)
            if l == len(o.tok) and l > best[0] and not busy:
                best = (l, o, None, True)
            for a in o.snapshots:
                if a.location <= l and a.location > best[0]:
                    best = (a.location, o, a, a is o.snapshots[-1] and not busy)
        location, o, a, in_place = best
        if o is None or location < self.min_len:
            y = Session(self.m)
            self.sessions.append(y)
            return y, "new"
        no_touch = transient and not o.transient
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

    def total_memory(self) -> int:
        seen: set = set()
        return sum(o.memory(seen) for o in self.sessions)


# ---------------------------------------------------------------- prompt reading


def step_forward(m: GemmaEngine, o: Session, ids: List[int], image=None, snapshot_places=(), chunk: int = 512,
           halt: Optional[Callable[[], bool]] = None, image_ends=()):
    """Reads the part of ids that is not in the session (from len(o.tok) to the end); takes snapshots at snapshot_places.

    image = (embeddings of the whole prompt (1, n, H), image token mask (n,)) - if the part to be read contains images.
    A chunk boundary never falls in the middle of an image (the image tokens attend to each other bidirectionally).
    Chunk size 512: measured (11,248 tokens): 512 -> 769 · 1,024 -> 747 · 2,048 -> 718 · single pass -> 578 tokens/s.
    Snapshot places do NOT split the chunk (2026-11-11): every forward pass has a fixed cost of ~40 ms (all the weights are read once); a separate pass of
    a few tokens for a checkpoint cost 45-170 ms per request. The snapshot is sliced from the cache after the chunk is read (Session.snapshot_slice).
    image_ends: end positions of the images in the prompt (to know how many images a snapshot covers).
    """
    begin, n = len(o.tok), len(ids)
    if begin >= n:
        return
    p = mx.array(ids)[None]
    embedding, mm = image if image is not None else (None, None)
    checkpoints = sorted(x for x in set(snapshot_places) if begin < x < n)
    s = begin
    while s < n:
        e = min(s + chunk, n)
        if mm is not None and e < n and mm[e - 1] and mm[e]:          # middle of an image: if the part before it is short, extend to the end of the block; if long, cut at the start of the block
            b = e
            while b > s and mm[b - 1]:
                b -= 1
            if b - s >= 256:
                e = b
            else:
                while e < n and mm[e]:
                    e += 1
        if n - e < 16:                                                # do not leave a separate pass of a few tokens at the end
            e = n
        if mm is not None and mm[s:e].any():
            g = m.hidden_image(embedding[:, s:e, :], mm[s:e], o.ct)
        else:
            g = m.hidden(p[:, s:e], o.ct)
        o.hidden = g[:, -1:, :]
        mx.eval(o.hidden, o.states())
        o.tok.extend(ids[s:e])
        for d in [x for x in checkpoints if s < x <= e]:
            if d == e and e - s == 1:
                o.take_snapshot()                                          # single-token chunk: take it as is, since the cache is in rotating layout
            elif d < n:
                o.snapshot_slice(d, e, g[:, d - 1 - s : d - s, :], sum(1 for x in image_ends if x <= d))
        s = e
        mx.clear_cache()
        if halt is not None and halt():
            raise Cancelled()


# ---------------------------------------------------------------- generation


def _shadow(session_cache):
    return [copy.copy(c) for c in session_cache]


def _process(original, shadows, n: int, m: int, previous: List[int]):
    """Feeds the first m of the n tokens appended to the shadows during verification into the real cache."""
    at = n - m
    for a, g, p in zip(original, shadows, previous):
        if isinstance(a, RotatingKVCache):      # a multi-token append lays entries out in time order: the trailing 'cut' entries are the rejected ones
            a.keys = g.keys if at == 0 else g.keys[..., : g.keys.shape[2] - at, :]
            a.values = g.values if at == 0 else g.values[..., : g.values.shape[2] - at, :]
            a._idx = a.keys.shape[2]
        else:                                    # KVCache: rejected entries stay beyond the offset; the next append overwrites them
            a.keys, a.values = g.keys, g.values
        a.offset = p + m


def _draft_block(m: GemmaEngine, ct, sampler: Sampler, si: int, hidden_norm, d: int, kv: int):
    """d draft tokens: (sequence (1,d), candidate ids [(1,K)]×d, probabilities [(1,K)]×d). For greedy the last two are None."""
    t = m.draft
    shared = m.lm.shared_kv_from_cache(ct)
    t.set_shared_kv(shared, kv, position=max(kv - 1, 0), kv_valid_len=kv)
    try:
        if sampler.is_greedy:
            return t.draft_block(si, hidden_norm, None, d + 1, _argmax, mx.int32, greedy=True), None, None
        location = mx.array([[max(kv - 1, 0)]])
        tok = mx.array([[si]], dtype=mx.int32)
        h = hidden_norm
        arr, q_cand, q_prob = [], [], []
        for _ in range(d):
            e = t._input_embed(tok) * t._input_embed_scale
            h, lg = t(mx.concatenate([e, h], axis=-1), shared, location)
            cand, p = sampler.dist(lg[:, -1, :])
            y = sampler.draw(cand, p)
            arr.append(y[None])
            q_cand.append(cand)
            q_prob.append(p)
            tok = y[None].astype(mx.int32)
        return mx.concatenate(arr, axis=1), q_cand, q_prob
    finally:
        t.release_shared_kv()


def generate(m: GemmaEngine, o: Session, max_n: int, sampler: Sampler, stats: Stats,
         halt: Optional[Callable[[], bool]] = None, constraint=None, no_draft: bool = False,
         trace: Optional[list] = None, force: Optional[Callable[[List[int]], Optional[int]]] = None) -> Iterator[List[int]]:
    """Generates from the end of the session; yields a new list of tokens each round.

    When done the session is left consistent: the cache holds the prompt + all generated tokens except the LAST one (the last token -
    usually the end marker - is read as part of the next prompt); o.hidden is the hidden state of the last slot.
    """
    ct = o.ct
    controller = m.depth_greedy if sampler.is_greedy else m.depth

    def pick(lg):
        if sampler.is_greedy:
            return mx.argmax(lg, axis=-1), None, None
        cand, p = sampler.dist(lg)
        return sampler.draw(cand, p), cand, p

    t_begin = time.perf_counter()
    lg = m.logit(o.hidden)[:, -1, :]
    if constraint is not None:
        lg = constraint.apply(lg)
    current, _, _ = pick(lg)
    mx.eval(current)
    si = current.item()
    if constraint is not None:
        constraint.consume([si], m.eos)
    generated = 1
    last_hidden = o.hidden               # hidden state of the last slot in the cache (anticipating si)
    finished = si in m.eos
    all_tok = [si]
    yield [si]

    try:
        while not finished and generated < max_n:
            if halt is not None and halt():
                stats.finish = "cancelled"
                break
            if constraint is not None and constraint.finished():
                # Schema complete (the closing `}` was chosen): only the end marker remains. A ~45 ms forward pass is not spent to produce it;
                # the last token stays out of the cache as usual (the session is consistent).
                finished = True
                break
            remaining = max_n - generated
            d = 0 if no_draft else max(0, min(controller.pick(), remaining - 1))
            forced = force(all_tok) if force is not None else None
            if forced is not None:
                d = 0
            mandatory = constraint.mandatory()[: remaining - 1] if (constraint is not None and forced is None and remaining > 1) else []
            t0 = time.perf_counter()

            if mandatory:
                # Sequence for which the schema leaves a single option (like a field name): no model decision is needed -> [current + forced] go straight into the main
                # cache in a single forward pass, then the following token is picked from the main model. The draft model and the shadow copy are not used (2026-11-11).
                g = m.hidden(mx.array([[si] + mandatory], dtype=mx.int32), ct)
                constraint.consume(mandatory, m.eos)
                y, _, _ = pick(constraint.apply(m.logit(g[:, -1:, :])[:, -1, :]))
                mx.eval(y, g)
                o.tok.append(si)
                o.tok.extend(mandatory)
                last_hidden = g[:, -1:, :]
                si = y.item()
                constraint.consume([si], m.eos)
                new, accept = mandatory + [si], 0
                stats.mandatory = getattr(stats, "mandatory", 0) + len(mandatory)
                if trace is not None:
                    trace.append((mandatory, len(mandatory), si))
            elif d == 0:
                g = m.hidden(mx.array([[si]], dtype=mx.int32), ct)
                lg = m.logit(g)[:, -1, :]
                if constraint is not None:
                    lg = constraint.apply(lg)
                y, _, _ = pick(lg)
                if forced is not None:
                    y = mx.array([forced])
                mx.eval(y, g)
                o.tok.append(si)
                last_hidden = g[:, -1:, :]
                si = y.item()
                if constraint is not None:
                    constraint.consume([si], m.eos)
                new, accept = [si], 0
            else:
                # 1) the draft model proposes d tokens (by looking at the keys/values of the main model's last layers)
                t, q_cand, q_prob = _draft_block(m, ct, sampler, si, m.draft_hidden(last_hidden), d, len(o.tok))
                # 2) the main model verifies [current, drafts] in one pass - on the cache's shadow
                fed = mx.concatenate([mx.array([[si]], dtype=mx.int32), t.astype(mx.int32)], axis=1)
                previous = [c.offset for c in ct]
                shadow = _shadow(ct)
                hv = m.hidden(fed, shadow)
                lg = m.logit(hv)[0]
                # Schema: unmasked candidates are computed first (the round stays a single evaluation); only the rows whose candidate violates the schema are masked and
                # recomputed (constraint.mask_needed_rows). In free-text fields no mask is ever built; the result is the same as the masked computation.
                if sampler.is_greedy:
                    tgt = mx.argmax(lg, axis=-1)
                    mx.eval(t, tgt)
                    tl, hl = t[0].tolist(), tgt.tolist()
                    if constraint is not None:
                        needed = constraint.mask_needed_rows(tl, [[h] for h in hl])
                        if needed:
                            rows = sorted(needed)
                            yh = mx.argmax(constraint.apply_mask(lg[mx.array(rows)], np.concatenate([needed[i] for i in rows], axis=0)), axis=-1).tolist()
                            for j, i in enumerate(rows):
                                hl[i] = int(yh[j])
                            stats.masked = getattr(stats, "masked", 0) + len(rows)
                    accept = 0
                    while accept < d and tl[accept] == hl[accept]:
                        accept += 1
                    following = hl[accept]
                else:
                    p_cand, p_prob = sampler.dist(lg)
                    qk, qp = mx.concatenate(q_cand, axis=0), mx.concatenate(q_prob, axis=0)
                    mx.eval(t, qk, qp, p_cand, p_prob)
                    tl = t[0].tolist()
                    pk, pp = np.array(p_cand), np.array(p_prob)
                    if constraint is not None:
                        # only candidates whose probability stays > 0 are looked at (those outside the nucleus cannot be selected anyway); rows needing a mask go in ONE evaluation
                        needed = constraint.mask_needed_rows(tl, [pk[i][pp[i] > 0] for i in range(len(pk))])
                        if needed:
                            rows = sorted(needed)
                            yk, yp = sampler.dist(constraint.apply_mask(lg[mx.array(rows)], np.concatenate([needed[i] for i in rows], axis=0)))
                            yk, yp = np.array(yk), np.array(yp)
                            for j, i in enumerate(rows):
                                pk[i], pp[i] = yk[j], yp[j]
                            stats.masked = getattr(stats, "masked", 0) + len(rows)
                    accept, following = sampler.accept_reject(tl, np.array(qk), np.array(qp), pk, pp)
                # if the end marker is among the accepted drafts, stop there (the end marker does not enter the cache)
                observed = accept
                for i in range(accept):
                    if tl[i] in m.eos:
                        accept, following = i, tl[i]
                        break
                new = tl[:accept] + [following]
                if constraint is not None:
                    # if the schema completed in the middle of this round (the closing `}` is one of the accepted drafts), what follows is discarded: since those rows
                    # are not masked, the main model can continue into free text (2026-11-13; see Constraint.consume_and_cut). Treated like an end marker:
                    # the closing token becomes "current", and only the ones before it are fed into the cache.
                    hold = constraint.consume_and_cut(new, m.eos)
                    if hold < len(new):
                        accept, following, new = hold - 1, new[hold - 1], new[:hold]
                        stats.schema_cuts = getattr(stats, "schema_cuts", 0) + 1
                if trace is not None:
                    trace.append((tl, accept, following))
                # 3) only [current + accepted] are fed into the main cache
                _process(ct, shadow, d + 1, accept + 1, previous)
                o.tok.append(si)
                o.tok.extend(tl[:accept])
                last_hidden = hv[:, accept : accept + 1, :]
                si = following
                stats.draft += d
                stats.accept += observed
                accept = observed

            stats.kind += 1
            if mandatory:
                controller.prev_d = -1         # the forced round does not enter the depth statistics; the time of the round that follows is not counted either (the format changed)
            else:
                controller.observe(d, accept, 1000 * (time.perf_counter() - t0))
            for i, b in enumerate(new):
                if b in m.eos:
                    new = new[: i + 1]
                    finished = True
                    break
            generated += len(new)
            all_tok.extend(new)
            if stats.kind % 64 == 0:
                mx.clear_cache()
            yield new
        if not finished and stats.finish == "stop":
            stats.finish = "length"
    finally:
        o.hidden = last_hidden
        mx.eval(o.hidden, o.states())
        stats.generated = generated
        stats.generate_secs = time.perf_counter() - t_begin
        o.latest = time.time()
