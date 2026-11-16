"""Produces the engine's model from Qwen's original weights (nvfp4, 4 bit, group 16).

Left uncompressed (bf16), the same recipe as Ollama's own model (2026-11-10, inferred from tensor sizes): the token embedding table
(`embed_tokens`) and the gate projections of the linear-attention layers (`linear_attn.in_proj_a`, `in_proj_b`; they produce the forget / write
rates of the recurrent state). Plain `mlx_vlm convert` quantizes these to 4 bit as well.

Usage: python tools/convert.py <original folder> <output folder>
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import importlib
importlib.import_module("mlx_vlm.convert"); C = sys.modules["mlx_vlm.convert"]
from mlx_vlm.models.qwen3_5 import qwen3_5 as Q

COMPRESS_NAMES = ("embed_tokens", ".in_proj_a", ".in_proj_b")
_original = Q.Model.quant_predicate            # property


def _rule(self):
    original = _original.fget(self)

    def rule(file_path, mod_name):
        if file_path.endswith(COMPRESS_NAMES):
            return False
        return original(file_path, mod_name) if original is not None else True

    return rule


Q.Model.quant_predicate = property(_rule)
source, tgt = sys.argv[1], sys.argv[2]
C.convert(source, tgt, quantize=True, q_group_size=16, q_bits=4, q_mode="nvfp4")
print("done:", tgt)
