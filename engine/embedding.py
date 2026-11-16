"""Embedding model: bge-m3 (XLM-RoBERTa large) — for memory and library searches.

Weights are mlx-community/bge-m3-mlx-fp16 (MIT). The output has the same definition as bge-m3 in Ollama:
the last hidden state of the [CLS] token, normalized to unit length (1024 dimensions).
"""

from __future__ import annotations

import json
import os
from typing import List

import mlx.core as mx
import mlx.nn as nn
import numpy as np

PATH = os.environ.get("FARABI_EMBED_MODEL") or os.path.join(os.environ.get("FARABI_MODELS") or os.path.expanduser("~/.farabi/models"), "bge-m3-mlx-fp16")
MAX_TOKENS = 8192


class _Layer(nn.Module):
    def __init__(self, h: int, heads: int, inter: int, eps: float):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, h)
        self.outp = nn.Linear(h, h)
        self.norm1 = nn.LayerNorm(h, eps=eps)
        self.inter = nn.Linear(h, inter)
        self.latest = nn.Linear(inter, h)
        self.norm2 = nn.LayerNorm(h, eps=eps)

    def __call__(self, x, attn_mask):
        B, L, H = x.shape
        d = H // self.heads

        def divide(t):
            return t.reshape(B, L, self.heads, d).transpose(0, 2, 1, 3)

        a = mx.fast.scaled_dot_product_attention(divide(self.q(x)), divide(self.k(x)), divide(self.v(x)), scale=d**-0.5, mask=attn_mask)
        a = a.transpose(0, 2, 1, 3).reshape(B, L, H)
        x = self.norm1(x + self.outp(a))
        return self.norm2(x + self.latest(nn.gelu(self.inter(x))))


class Embedder:
    def __init__(self, file_path: str = PATH):
        from tokenizers import Tokenizer

        c = json.load(open(os.path.join(file_path, "config.json")))
        self.h, self.padding = c["hidden_size"], c["pad_token_id"]
        eps = c.get("layer_norm_eps", 1e-5)
        self.tok = Tokenizer.from_file(os.path.join(file_path, "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=min(MAX_TOKENS, c["max_position_embeddings"] - 2))
        w = mx.load(os.path.join(file_path, "model.safetensors"))
        self.words = w["embeddings.word_embeddings.weight"]
        self.location = w["embeddings.position_embeddings.weight"]
        self.kind = w["embeddings.token_type_embeddings.weight"][0]
        self.norm = nn.LayerNorm(self.h, eps=eps)
        self.norm.weight, self.norm.bias = w["embeddings.LayerNorm.weight"], w["embeddings.LayerNorm.bias"]
        self.layer_list: List[_Layer] = []
        for i in range(c["num_hidden_layers"]):
            k = _Layer(self.h, c["num_attention_heads"], c["intermediate_size"], eps)
            o = f"encoder.layer.{i}."
            for tgt, name in ((k.q, "attention.self.query"), (k.k, "attention.self.key"), (k.v, "attention.self.value"),
                              (k.outp, "attention.output.dense"), (k.inter, "intermediate.dense"), (k.latest, "output.dense"),
                              (k.norm1, "attention.output.LayerNorm"), (k.norm2, "output.LayerNorm")):
                tgt.weight, tgt.bias = w[o + name + ".weight"], w[o + name + ".bias"]
            self.layer_list.append(k)
        mx.eval(self.words, self.location, [k.parameters() for k in self.layer_list])

    def _forward(self, ids: np.ndarray, lengths: np.ndarray) -> mx.array:
        B, L = ids.shape
        x = mx.array(ids)
        valid = np.arange(L)[None, :] < lengths[:, None]
        # XLM-R: position numbers start after the padding id (2, 3, ...); padding slots stay at the padding position
        location = np.where(valid, np.arange(L)[None, :] + self.padding + 1, self.padding)
        h = self.words[x] + self.location[mx.array(location)] + self.kind
        h = self.norm(h)
        attn_mask = None
        if not valid.all():
            attn_mask = mx.array(np.where(valid, 0.0, -1e4).astype(np.float16))[:, None, None, :].astype(h.dtype)
        for k in self.layer_list:
            h = k(h, attn_mask)
        v = h[:, 0, :].astype(mx.float32)
        return v / mx.maximum(mx.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    def embed(self, texts: List[str], batch: int = 16) -> List[List[float]]:
        """A 1024-dimensional unit vector for each text. Texts of similar length are processed together."""
        code = [self.tok.encode(t or " ").ids for t in texts]
        order = sorted(range(len(code)), key=lambda i: len(code[i]))
        outputs = [None] * len(code)
        for a in range(0, len(order), batch):
            grp = order[a : a + batch]
            L = max(len(code[i]) for i in grp)
            ids = np.full((len(grp), L), self.padding, dtype=np.int32)
            for r, i in enumerate(grp):
                ids[r, : len(code[i])] = code[i]
            v = self._forward(ids, np.array([len(code[i]) for i in grp]))
            mx.eval(v)
            for r, i in enumerate(grp):
                outputs[i] = v[r].tolist()
        self.last_token = sum(len(k) for k in code)
        return outputs
