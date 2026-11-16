"""Farabi engine — core.

Decodes Qwen 3.8 27B (mlx-lm) speculatively with its own MTP head: the head predicts the next
few tokens, the main model verifies all of them in ONE forward pass, and the ones that hold are
accepted. Same design as Ollama's mlxrunner/mtp.go + speculate.go; it depends on neither
Ollama's code nor its files.

Written against mlx-lm 0.32.2 + mlx 0.32.4 (GatedDeltaNet.__call__ is a copy from that version).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten
from mlx_lm import load as mlx_load
from mlx_lm.models import qwen3_5, qwen3_next
from mlx_lm.models.base import (
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

from .depth import Depth
from .sampling import Sampler


# ---------------------------------------------------------------- GDN recording
# The state of linear-attention (GatedDeltaNet) layers is recurrent: rejected draft
# tokens cannot be "erased" from the state. During the verification pass we save each
# layer's inputs (q, k, v, a, b), the previous state and the convolution input;
# on rollback we re-run the kernel for only the first n accepted tokens
# (48 small kernel calls; the MLP and projections are not recomputed).


def _gdn_call(self, inputs, mask=None, cache=None):
    B, S, _ = inputs.shape

    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
        )

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    n_keep = self.conv_kernel_size - 1
    if cache is not None:
        cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(self.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]

    state = cache[1] if cache else None
    q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)

    if cache is not None and getattr(cache, "record_on", False):
        cache.record = (conv_input, q, k, v, a, b, state, self)

    out, state = gated_delta_update(
        q, k, v, a, b, self.A_log, self.dt_bias, state, mask, use_kernel=True
    )

    if cache is not None:
        cache[1] = state
        cache.advance(S)

    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))


qwen3_5.GatedDeltaNet.__call__ = _gdn_call


def _gdn_rollback(cache, n: int):
    """Keeps only the first n tokens of the recorded verification pass."""
    conv_input, q, k, v, a, b, state0, layer = cache.record
    n_keep = layer.conv_kernel_size - 1
    cache[0] = mx.contiguous(conv_input[:, n : n + n_keep, :])
    _, st = gated_delta_update(
        q[:, :n], k[:, :n], v[:, :n], a[:, :n], b[:, :n],
        layer.A_log, layer.dt_bias, state0, None, use_kernel=True,
    )
    cache[1] = st


# ---------------------------------------------------------------- position
# Full-attention layers (16 in the main model, 1 in the MTP head) take the position from
# the cache length by default. With an image prompt the positions are 3-dimensional and the
# text after the image differs from its index in the sequence by `shift`; the engine
# sets this context before every forward pass.


class _Location:
    arr = None      # (3,1,L) explicit position ids (while reading an image prompt)
    shift = 0        # offset added to the cache length (text after the image)
    mrope = None     # mlx-vlm Qwen3_5RotaryEmbedding (only when vision is loaded)


LOCATION = _Location()


def _attn_call(self, x, mask=None, cache=None):
    B, L, D = x.shape
    q_proj_output = self.q_proj(x)
    queries, gate = mx.split(
        q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
    )
    gate = gate.reshape(B, L, -1)
    keys, values = self.k_proj(x), self.v_proj(x)
    queries = self.q_norm(queries).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

    if LOCATION.arr is not None:
        queries, keys = LOCATION.mrope.apply_rotary(queries, keys, LOCATION.arr, unsqueeze_dim=1)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
    elif cache is not None:
        queries = self.rope(queries, offset=cache.offset + LOCATION.shift)
        keys = self.rope(keys, offset=cache.offset + LOCATION.shift)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = self.rope(queries)
        keys = self.rope(keys)

    output = scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=self.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output * mx.sigmoid(gate))


qwen3_next.Qwen3NextAttention.__call__ = _attn_call


# ---------------------------------------------------------------- MTP head


class MTPHead(nn.Module):
    """Qwen's multi-token prediction head: a single full-attention layer.

    Input at slot S = [norm(embedding(token S+1)), norm(the main model's
    hidden state at S)] -> fc -> layer -> norm. Passed through lm_head, the output gives S+2.
    """

    def __init__(self, args):
        super().__init__()
        H = args.hidden_size
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layers = [
            qwen3_5.DecoderLayer(args, layer_idx=args.full_attention_interval - 1)
        ]
        self.norm = nn.RMSNorm(H, eps=args.rms_norm_eps)

    def __call__(self, embedding, hidden, cache):
        x = self.fc(
            mx.concatenate(
                [self.pre_fc_norm_embedding(embedding), self.pre_fc_norm_hidden(hidden)],
                axis=-1,
            )
        )
        mask = create_attention_mask(x, cache)
        return self.norm(self.layers[0](x, mask=mask, cache=cache))


def mtp_load(args, file_path: str, compress=None) -> MTPHead:
    """Loads the raw bf16 mtp.* tensors taken from the original Qwen repository."""
    raw = mx.load(file_path)
    weights = {}
    for name, t in raw.items():
        name = name[len("mtp."):]
        if t.ndim == 1:  # in the original repository the norm weights are stored as (w - 1)
            t = t + 1.0
        weights[name] = t
    begin = MTPHead(args)
    begin.load_weights(list(weights.items()), strict=True)
    begin.set_dtype(mx.bfloat16)
    if compress == "nvfp4":      # same format as the main model
        nn.quantize(begin, group_size=16, bits=4, mode="nvfp4")
    elif compress:               # integer: affine bit count
        nn.quantize(begin, group_size=64, bits=int(compress))
    mx.eval(begin.parameters())
    begin.eval()
    return begin


# ---------------------------------------------------------------- engine


@dataclass
class Result:
    tokens: List[int] = field(default_factory=list)
    kind: int = 0
    draft: int = 0
    accept: int = 0
    prompt_tok: int = 0
    prompt_secs: float = 0.0
    generate_secs: float = 0.0
    depths: List[int] = field(default_factory=list)

    @property
    def speed(self) -> float:
        return len(self.tokens) / max(self.generate_secs, 1e-9)


# The chat tokenizer STAYS IN THE PROCESS even after the model is unloaded (the counterpart of gemma._PERSISTENT for Gemma). Measured:
# of a 1.76 s warm load, 0.55 s is this tokenizer, 0.68 s is the vision processor (vision.py), and the weights take 0.47 s. It is a CPU object and carries
# no model-bound state; it is rebuilt if the config files in the model folder change (signature). FARABI_QWEN_PERSIST=0 falls back to the old path (mlx_lm.load).
# Cost: about 340 MB of RAM (both together) -> released GEMMA_PERSIST_SECS (30 min) after Qwen is unloaded (server.Worker._release_persistent_qwen).
_PERSISTENT: dict = {}        # model path -> (signature, tokenizer)


def config_signature(model_path: str) -> tuple:
    sig = []
    for name in sorted(os.listdir(model_path)):
        if not name.endswith(".safetensors"):
            try:
                st = os.stat(os.path.join(model_path, name))
                sig.append((name, st.st_mtime_ns, st.st_size))
            except OSError:
                pass
    return tuple(sig)


def _model_and_tok(model_path: str):
    """The SAME objects as `mlx_lm.load(model_path)`; the tokenizer is built once per process. Returns: (model, tokenizer, whether it came from the kept copy)."""
    if os.environ.get("FARABI_QWEN_PERSIST", "1") == "0":
        return (*mlx_load(model_path), False)
    from mlx_lm.utils import _download, load_model, load_tokenizer

    file_path = _download(model_path)
    model, config = load_model(file_path, False, model_config=None, trust_remote_code=False)
    signature = config_signature(model_path)
    k = _PERSISTENT.get(model_path)
    stored = k is not None and k[0] == signature
    if not stored:
        k = (signature, load_tokenizer(file_path, None, eos_token_ids=config.get("eos_token_id", None)))
        _PERSISTENT.clear()
        _PERSISTENT[model_path] = k
    return model, k[1], stored


class Engine:
    def __init__(self, model_path: str, mtp_path: Optional[str] = None, mtp_compress="nvfp4", vision: bool = False):
        self.model, self.tok, self.stored_tok = _model_and_tok(model_path)
        self.vision = None
        if vision:
            from .vision import VisionEncoder
            self.vision = VisionEncoder(model_path)
            LOCATION.mrope = self.vision.mrope
        self.lm = self.model.language_model
        self.body = self.lm.model
        self.args = self.lm.args
        self.mtp = mtp_load(self.args, mtp_path, mtp_compress) if mtp_path else None
        self.eos = set(self.tok.eos_token_ids)
        # conversion 3 model (tools/convert3.py): the lm_head weight is stored scaled up by 2^k; compensated here with a single multiply (a power of two -> exact)
        try:
            k = json.load(open(os.path.join(model_path, "config.json"))).get("farabi_lm_head_shift")
        except Exception:
            k = None
        self._lm_scale = (2.0 ** -int(k)) if k else None
        self.depth = Depth()   # depth state learned across requests

    # -- helpers
    def _hidden(self, fed, cache):
        return self.body(fed, cache=cache)

    def _logit(self, hidden):
        return self.lm.lm_head(hidden * self._lm_scale if self._lm_scale else hidden)

    def _embed(self, token):
        return self.body.embed_tokens(token)

    def _states(self, caches):
        d = []
        for c in caches:
            if isinstance(c, KVCache):
                if c.keys is not None:
                    d += [c.keys, c.values]
            else:
                d += [x for x in c.cache if x is not None]
        return d

    def session_cache(self):
        return self.lm.make_cache(), ([KVCache()] if self.mtp else None)

    # -- prompt reading
    def read_prompt(self, prompt: List[int], ct, cm, chunk: int = 2048, image=None):
        """Has the main model read the prompt; also fills the MTP head's cache.

        image = (embeddings (1,n,H), position (3,1,n), shift) — for an image prompt.
        Returns: the hidden state of the last slot (1,1,H).
        """
        p = mx.array(prompt)[None]
        n = p.shape[1]
        embedding = location = None
        if image is not None:
            embedding, location, _ = image
        latest = None
        try:
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                if embedding is None:
                    g = self._hidden(p[:, s:e], ct)
                else:
                    LOCATION.arr = location[:, :, s:e]
                    g = self.body(p[:, s:e], cache=ct, input_embeddings=embedding[:, s:e, :])
                outputs = [g[:, -1:, :]]
                if cm is not None:
                    # slot S pair = (token S+1, hidden S); the last slot's partner does not exist yet
                    e2 = min(e, n - 1)
                    if e2 > s:
                        if embedding is None:
                            eg = self._embed(p[:, s + 1 : e2 + 1])
                        else:
                            eg = embedding[:, s + 1 : e2 + 1, :]
                            LOCATION.arr = location[:, :, s:e2]
                        h = self.mtp(eg, g[:, : e2 - s, :], cm[0])
                        outputs.append(h[:, -1:, :])
                mx.eval(outputs, self._states(ct), self._states(cm or []))
                latest = outputs[0]
                mx.clear_cache()
        finally:
            LOCATION.arr = None
        return latest

    # -- plain (draft-free) decoding
    def normal(self, prompt: List[int], max_n: int, sampler: Optional[Sampler] = None, image=None) -> Result:
        sampler = sampler or Sampler()
        LOCATION.shift = image[2] if image is not None else 0
        r = Result(prompt_tok=len(prompt))
        ct, _ = self.session_cache()

        def pick(hidden):
            lg = self._logit(hidden)[:, -1, :]
            if sampler.is_greedy:
                return mx.argmax(lg, axis=-1)
            return sampler.draw(*sampler.dist(lg))

        t = time.perf_counter()
        y = pick(self.read_prompt(prompt, ct, None, image=image))
        mx.eval(y)
        r.prompt_secs = time.perf_counter() - t
        t = time.perf_counter()
        while True:
            b = y.item()
            r.tokens.append(b)
            r.kind += 1
            if b in self.eos or len(r.tokens) >= max_n:
                break
            y = pick(self._hidden(y[None], ct))
            mx.eval(y, self._states(ct))
        r.generate_secs = time.perf_counter() - t
        return r

    # -- speculative decoding with MTP
    def generate(self, prompt: List[int], max_n: int, depth=None, sampler: Optional[Sampler] = None, image=None) -> Iterator[List[int]]:
        """Yields a list of tokens per round; the last item is a Result object."""
        assert self.mtp is not None
        sampler = sampler or Sampler()
        LOCATION.shift = image[2] if image is not None else 0
        r = Result(prompt_tok=len(prompt))
        ct, cm = self.session_cache()
        gdn = [c for c in ct if not isinstance(c, KVCache)]
        kv = [c for c in ct if isinstance(c, KVCache)]
        if depth is None:
            depth = self.depth
        controller = depth if isinstance(depth, Depth) else None
        fixed = None if controller else depth

        def pick(lg):  # (1,V) logits -> (token (1,), candidate ids, probabilities)
            if sampler.is_greedy:
                return mx.argmax(lg, axis=-1), None, None
            cand, p = sampler.dist(lg)
            return sampler.draw(cand, p), cand, p

        t = time.perf_counter()
        hidden_last = self.read_prompt(prompt, ct, cm, image=image)
        current, _, _ = pick(self._logit(hidden_last)[:, -1, :])  # (1,)
        mx.eval(current)
        r.prompt_secs = time.perf_counter() - t

        t = time.perf_counter()
        first = current.item()
        r.tokens.append(first)
        yield [first]
        # pending pairs: (token, hidden) rows to be written to the head cache
        pending_tok = current[None]          # (1,1)
        pending_hid = hidden_last              # (1,1,H)
        finished = first in self.eos

        while not finished and len(r.tokens) < max_n:
            remaining = max_n - len(r.tokens)
            d = controller.pick() if controller else fixed
            d = max(0, min(d, remaining - 1))
            t0 = time.perf_counter()

            if d == 0:
                # plain step; the head cache grows with the pending pairs (the write is deferred)
                g = self._hidden(current[None], ct)
                y, _, _ = pick(self._logit(g)[:, -1, :])
                mx.eval(y, g, self._states(ct))
                pending_tok = mx.concatenate([pending_tok, y[None]], axis=1)
                pending_hid = mx.concatenate([pending_hid, g], axis=1)
                current = y
                new = [y.item()]
                accept = 0
            else:
                # 1) write the pending pairs in a single pass; the last row yields the first draft
                h = self.mtp(self._embed(pending_tok), pending_hid, cm[0])[:, -1:, :]
                draft, q_cand, q_prob = [], [], []
                for i in range(d):
                    if i > 0:
                        h = self.mtp(self._embed(draft[-1][None]), h, cm[0])
                    y, cand, p = pick(self._logit(h)[:, -1, :])
                    draft.append(y)
                    if cand is not None:
                        q_cand.append(cand)
                        q_prob.append(p)
                draft_arr = mx.stack(draft, axis=1)  # (1,d)

                # 2) main model: [current, drafts] in a single pass
                for c in gdn:
                    c.record_on = True
                fed = mx.concatenate([current[None], draft_arr], axis=1)
                g = self._hidden(fed, ct)  # (1,d+1,H)
                lg = self._logit(g)[0]      # (d+1,V)
                for c in gdn:
                    c.record_on = False
                if sampler.is_greedy:
                    tgt = mx.argmax(lg, axis=-1)
                    mx.eval(draft_arr, tgt, g, self._states(ct), self._states(cm))
                    tl = draft_arr[0].tolist()
                    hl = tgt.tolist()
                    accept = 0
                    while accept < d and tl[accept] == hl[accept]:
                        accept += 1
                    following = hl[accept]
                else:
                    p_cand, p_prob = sampler.dist(lg)
                    qk, qp = mx.concatenate(q_cand, axis=0), mx.concatenate(q_prob, axis=0)
                    mx.eval(draft_arr, qk, qp, p_cand, p_prob, g, self._states(ct), self._states(cm))
                    tl = draft_arr[0].tolist()
                    accept, following = sampler.accept_reject(
                        tl, np.array(qk), np.array(qp), np.array(p_cand), np.array(p_prob)
                    )
                new = tl[:accept] + [following]

                # 3) roll back: the main model cache keeps only [current + accepted]
                if accept < d:
                    for c in kv:
                        c.trim(d - accept)
                    for c in gdn:
                        _gdn_rollback(c, accept + 1)
                for c in gdn:
                    c.record = None
                # head cache: speculative writes (d-1 slots) are removed
                if d > 1:
                    cm[0].trim(d - 1)
                # new pending pairs: the accepted drafts + the next token
                pending_tok = mx.array([new])
                pending_hid = g[:, : accept + 1, :]
                current = mx.array([new[-1]])
                r.draft += d
                r.accept += accept

            r.kind += 1
            r.depths.append(d)
            for i, b in enumerate(new):
                if b in self.eos:
                    new = new[: i + 1]
                    finished = True
                    break
            r.tokens += new
            if controller:
                controller.observe(d, accept, 1000 * (time.perf_counter() - t0))
            yield new

        r.tokens = r.tokens[:max_n]
        r.generate_secs = time.perf_counter() - t
        yield r
