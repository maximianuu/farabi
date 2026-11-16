"""Produces the engine's model with calibrated (error-compensated, GPTQ) nvfp4 (experimental, 2026-11-10..11).

Runs the original Qwen layer by layer over calibration texts (tools/calibration.py): in each layer the second moment of the real inputs of the linear
matrices is collected, the matrices are quantized with tools/gptq.py, and the layer's output is recomputed with the QUANTIZED weights and passed to the next
layer (later layers see the error of earlier ones and compensate for it). The 2^k shift and folding rules are the same as in convert3.py;
the folded tensors (norms, in_proj_a/b) are computed from the original values, and the base model is used only for layout and untouched tensors.

Usage: python tools/convert4.py <original folder> <base MLX model> <output folder> [only the first N layers: dry run, writes nothing to disk]
"""
import glob, json, os, shutil, sys, time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gptq as G
import calibration
import nvfp4 as N
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3_5 import create_ssm_mask
from mlx_lm.utils import load_model
from transformers import AutoTokenizer

orig, base_dir, tgt = sys.argv[1:4]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
MARGIN = 0.5                      # values that grow during compensation must not exceed the scale upper limit (448): one bit of headroom
F32, BF16 = mx.float32, mx.bfloat16
t0 = time.time()

tok = AutoTokenizer.from_pretrained(base_dir)
D = calibration.sequences(tok)
print(f"calibration: {len(D)} sequences, {sum(map(len, D))} tokens (longest {max(map(len, D))})", flush=True)

model, _ = load_model(Path(orig), lazy=True)
lm = model.language_model
g = lm.model
Hs = [g.embed_tokens(mx.array(d)[None]).astype(F32) for d in D]
mx.eval(Hs)

RECORD = {}                     # id(module) → [XᵀX sum, row count]
_old = nn.Linear.__call__


def _call(self, x):
    r = RECORD.get(id(self))
    if r is not None:
        X = x.reshape(-1, x.shape[-1]).astype(F32)
        r[0] = X.T @ X if r[0] is None else r[0] + X.T @ X
        r[1] += X.shape[0]
    return _old(self, x)


nn.Linear.__call__ = _call

A, file = {}, {}
if LIMIT is None:
    os.makedirs(tgt, exist_ok=False)
    for f in sorted(glob.glob(os.path.join(base_dir, "model-*.safetensors"))):
        for k, v in mx.load(f).items():
            A[k] = v; file[k] = os.path.basename(f)
LM = "language_model.model.layers."
report = []


def walk(root, file_path):
    for p in file_path.split("."):
        root = getattr(root, p)
    return root


def kmax(m):
    return N.choose_shift(m.weight, MARGIN)


def divide(total, a_cap, b_ust):
    enabled = a_cap + b_ust - total
    assert enabled >= 0, (total, a_cap, b_ust)
    a = a_cap - enabled // 2
    return a, total - a


def compress(name, m, H, n, k, entry_key):
    """m.weight (original, f32) → GPTQ; writes the dequantized (true-scale) weight into the module and the quantized one into the output dict."""
    W = m.weight.astype(F32)
    Hn = H / n
    w, s = G.gptq(W, Hn, k)
    Wq = N.dequant(w, s) * (2.0 ** -k)
    h_g = G.output_err(W, Wq, Hn) ** 0.5
    w0, s0 = N.compress(W, k)
    h_d = G.output_err(W, N.dequant(w0, s0) * (2.0 ** -k), Hn) ** 0.5
    m.weight = Wq
    mx.eval(m.weight)
    if A:
        assert A[entry_key + ".weight"].shape == w.shape and A[entry_key + ".scales"].shape == s.shape, name
        A[entry_key + ".weight"], A[entry_key + ".scales"] = w, s
    report.append({"module": name, "k": k, "output_error_plain": h_d, "output_error_gptq": h_g})
    mx.clear_cache()
    return h_d, h_g


def save(entry_key, value):
    if A:
        assert A[entry_key].shape == value.shape, entry_key
        A[entry_key] = value.astype(A[entry_key].dtype)


n_layers = len(g.layers) if LIMIT is None else LIMIT
for i in range(n_layers):
    lay = g.layers[i]
    lay.set_dtype(F32)
    p = f"{LM}{i}."
    if lay.is_linear:
        la = lay.linear_attn
        groups = [(la.in_proj_qkv, [("linear_attn.in_proj_qkv", la.in_proj_qkv), ("linear_attn.in_proj_z", la.in_proj_z)]), (la.out_proj, [("linear_attn.out_proj", la.out_proj)])]
        kg = min(kmax(la.in_proj_qkv), kmax(la.in_proj_z)); kc = kmax(la.out_proj)
        ks = {"linear_attn.in_proj_qkv": kg, "linear_attn.in_proj_z": kg, "linear_attn.out_proj": kc}
    else:
        sa = lay.self_attn
        groups = [(sa.q_proj, [("self_attn.q_proj", sa.q_proj), ("self_attn.k_proj", sa.k_proj), ("self_attn.v_proj", sa.v_proj)]), (sa.o_proj, [("self_attn.o_proj", sa.o_proj)])]
        kg = min(kmax(sa.q_proj), kmax(sa.k_proj)); kv, ko = divide(kg, kmax(sa.v_proj), kmax(sa.o_proj))
        ks = {"self_attn.q_proj": kg, "self_attn.k_proj": kg, "self_attn.v_proj": kv, "self_attn.o_proj": ko}
    kn = kmax(lay.mlp.gate_proj); ku, kd = divide(kn, kmax(lay.mlp.up_proj), kmax(lay.mlp.down_proj))
    ks.update({"mlp.gate_proj": kn, "mlp.up_proj": ku, "mlp.down_proj": kd})
    groups += [(lay.mlp.gate_proj, [("mlp.gate_proj", lay.mlp.gate_proj), ("mlp.up_proj", lay.mlp.up_proj)]), (lay.mlp.down_proj, [("mlp.down_proj", lay.mlp.down_proj)])]
    # original values to be folded in (before the weights change)
    if lay.is_linear:
        save(p + "linear_attn.in_proj_a.weight", la.in_proj_a.weight * (2.0 ** kg)); save(p + "linear_attn.in_proj_b.weight", la.in_proj_b.weight * (2.0 ** kg))
        save(p + "linear_attn.norm.weight", la.norm.weight * (2.0 ** -kc))
    save(p + "input_layernorm.weight", lay.input_layernorm.weight * (2.0 ** -kg))
    save(p + "post_attention_layernorm.weight", lay.post_attention_layernorm.weight * (2.0 ** -kn))

    # 1) input statistics (this layer with the original weights, inputs coming from the already-quantized previous layers)
    for measured, _ in groups:
        RECORD[id(measured)] = [None, 0]
    for h in Hs:
        y = lay(h, mask=(create_ssm_mask(h, None) if lay.is_linear else create_attention_mask(h, None)), cache=None)
        mx.eval(y, [r[0] for r in RECORD.values()])
    t1 = time.time()
    # 2) quantize
    hd, hg = [], []
    for measured, members in groups:
        H, n = RECORD.pop(id(measured))
        for file_path, m in members:
            a, b = compress(f"layers.{i}.{file_path}", m, H, n, ks[file_path], p + file_path)
            hd.append(a); hg.append(b)
        del H
    RECORD.clear()
    # 3) recompute the output with the quantized weights
    Hs = [lay(h, mask=(create_ssm_mask(h, None) if lay.is_linear else create_attention_mask(h, None)), cache=None) for h in Hs]
    mx.eval(Hs)
    g.layers[i] = None; del lay; mx.clear_cache()
    print(f"layer {i + 1}/{n_layers} · {time.time() - t0:.0f} s (quantization {time.time() - t1:.0f} s) · output error plain {100 * sum(hd) / len(hd):.2f}% → compensated {100 * sum(hg) / len(hg):.2f}%"
          f" · peak memory {mx.get_peak_memory() / 1e9:.0f} GB", flush=True)

if LIMIT is not None:
    print("dry run done (nothing written to disk)"); sys.exit(0)

# lm_head: input = output of the final norm
g.norm.set_dtype(F32); lm.lm_head.set_dtype(F32)
RECORD[id(lm.lm_head)] = [None, 0]
for h in Hs:
    y = lm.lm_head(g.norm(h)); mx.eval(y[:, :1, :8], RECORD[id(lm.lm_head)][0])
H, n = RECORD.pop(id(lm.lm_head))
klm = kmax(lm.lm_head)
a, b = compress("lm_head", lm.lm_head, H, n, klm, "language_model.lm_head")
print(f"lm_head · output error plain {100 * a:.2f}% → compensated {100 * b:.2f}% · {time.time() - t0:.0f} s", flush=True)

for name in sorted(set(file.values())):
    mx.save_safetensors(os.path.join(tgt, name), {k: v for k, v in A.items() if file[k] == name}, metadata={"format": "mlx"})
    print("written:", name, flush=True)
for f in os.listdir(base_dir):
    if not f.endswith(".safetensors") and os.path.isfile(os.path.join(base_dir, f)):
        shutil.copy(os.path.join(base_dir, f), os.path.join(tgt, f))
cfg = json.load(open(os.path.join(tgt, "config.json")))
cfg["farabi_lm_head_shift"] = klm
cfg["farabi_compression"] = "nvfp4 · 2^k shift + error-compensated (GPTQ) rounding, calibration: tools/calibration.py (tools/convert4.py)"
json.dump(cfg, open(os.path.join(tgt, "config.json"), "w"), indent=2, ensure_ascii=False)
json.dump(report, open(os.path.join(tgt, "compression-report.json"), "w"), indent=0)
hd = [x["output_error_plain"] for x in report]; hg = [x["output_error_gptq"] for x in report]
print(f"done: {tgt} · {len(report)} matrices · output error avg plain {100 * sum(hd) / len(hd):.2f}% → compensated {100 * sum(hg) / len(hg):.2f}% · {time.time() - t0:.0f} s")
