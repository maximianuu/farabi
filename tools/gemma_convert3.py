"""Produces Gemma 4 31B with improved nvfp4 (calibration-free, 2026-11-11). Source: the original weights + the stock MLX conversion (for layout).

Why: the library's ready-made conversion (`mlx_vlm.convert -q --q-mode nvfp4`) pushes block scales into the subnormal region of E4M3 (see tools/nvfp4.py);
on Gemma the visible result was in long-form generation: instead of picking the single `<strong>` token the model drifted onto the multi-piece path and wrote
`<strong la="…">` (the 8-bit conversion is clean; the stock 4-bit is broken — measured). The embedding table (also the output head) is not quantized: Ollama's package leaves it as is too.

What is done: every quantized matrix is scaled up by 2^k and the block scales are chosen with least squares (tools/nvfp4.py). On Gemma the compensation is easier than on Qwen,
because the output of every projection goes straight through an RMS norm (independent of scale):
  q_proj → q_norm · k_proj → k_norm (+ v_norm on k=v layers) · v_proj → v_norm · o_proj → post_attention_layernorm     : k FREE
  up_proj, down_proj → (the product is linear) → post_feedforward_layernorm                                            : k FREE
  gate_proj (its output enters GELU)                                                                                   : pre_feedforward_layernorm ← 2^-k
The eps of the norms must also be scaled with the input's scale so the function stays EXACTLY the same: eps ← eps · 2^(2·scale exponent). The exponents are written to the
config.json → "farabi_norm_eps_exp" field; the engine applies them on load (engine/gemma.py). The stock library does not know this field (difference: only eps's share, small).

Usage: python tools/gemma_convert3.py <original folder> <base MLX model> <output folder>
"""
import glob, json, os, shutil, sys, time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvfp4 as N

orig, base_dir, tgt = sys.argv[1:4]
os.makedirs(tgt, exist_ok=False)
t0 = time.time()
W = N.orig_weights(orig)
chunks = sorted(glob.glob(os.path.join(base_dir, "model-*.safetensors")))
file, A = {}, {}
for f in chunks:
    for k, v in mx.load(f).items():
        A[k] = v; file[k] = os.path.basename(f)
LM, ORIG = "language_model.model.layers.", "model.language_model.layers."
error_list, eps_exp = [], {}


def write_out(i, file_path):
    """Quantizes the layers.i.<path> matrix with its own best shift; returns the shift."""
    w0 = W[f"{ORIG}{i}.{file_path}.weight"].astype(mx.float32)
    k = N.choose_shift(w0)
    w, s = N.compress(w0, k)
    hk = f"{LM}{i}.{file_path}"
    assert A[hk + ".weight"].shape == w.shape and A[hk + ".scales"].shape == s.shape, hk
    h = (mx.linalg.norm((N.dequant(w, s) * 2.0 ** -k - w0).reshape(-1)) / mx.linalg.norm(w0.reshape(-1))).item()
    h0 = (mx.linalg.norm((mx.dequantize(A[hk + ".weight"], A[hk + ".scales"], group_size=16, bits=4, mode="nvfp4").astype(mx.float32) - w0).reshape(-1))
          / mx.linalg.norm(w0.reshape(-1))).item()
    A[hk + ".weight"], A[hk + ".scales"] = w, s
    mx.eval(w, s)
    error_list.append((f"layers.{i}.{file_path}", k, h, h0))
    mx.clear_cache()
    return k


def mul_pow2(entry_key, us):
    v = A[entry_key]
    A[entry_key] = (v.astype(mx.float32) * (2.0 ** us)).astype(v.dtype)       # power of two: exact in bf16


n = 1 + max(int(k[len(LM):].split(".")[0]) for k in A if k.startswith(LM))
for i in range(n):
    kq = write_out(i, "self_attn.q_proj"); eps_exp[f"layers.{i}.self_attn.q_norm"] = 2 * kq
    kk = write_out(i, "self_attn.k_proj"); eps_exp[f"layers.{i}.self_attn.k_norm"] = 2 * kk
    if f"{LM}{i}.self_attn.v_proj.weight" in A:
        kv = write_out(i, "self_attn.v_proj")
    else:
        kv = kk                                  # k = v layer: the values are the raw output of k_proj
    eps_exp[f"layers.{i}.self_attn.v_norm"] = 2 * kv
    ko = write_out(i, "self_attn.o_proj"); eps_exp[f"layers.{i}.post_attention_layernorm"] = 2 * ko
    kn = write_out(i, "mlp.gate_proj"); mul_pow2(f"{LM}{i}.pre_feedforward_layernorm.weight", -kn)
    ku = write_out(i, "mlp.up_proj"); kd = write_out(i, "mlp.down_proj")
    eps_exp[f"layers.{i}.post_feedforward_layernorm"] = 2 * (ku - kn + kd)
    if i % 6 == 5:
        latest = error_list[-40:]
        print(f"layer {i + 1}/{n} · {time.time() - t0:.0f} s · relative weight error: stock conversion {100 * sum(x[3] for x in latest) / len(latest):.2f}% → this {100 * sum(x[2] for x in latest) / len(latest):.2f}%", flush=True)

for f in chunks:
    name = os.path.basename(f)
    mx.save_safetensors(os.path.join(tgt, name), {k: v for k, v in A.items() if file[k] == name}, metadata={"format": "mlx"})
    print("written:", name, flush=True)
for f in os.listdir(base_dir):
    if not f.endswith(".safetensors") and os.path.isfile(os.path.join(base_dir, f)):
        shutil.copy(os.path.join(base_dir, f), os.path.join(tgt, f))
cfg = json.load(open(os.path.join(tgt, "config.json")))
cfg["farabi_norm_eps_exp"] = eps_exp
cfg["farabi_compression"] = "nvfp4 · 2^k shift + least-squares block scale, embedding table bf16 (tools/gemma_convert3.py)"
json.dump(cfg, open(os.path.join(tgt, "config.json"), "w"), indent=2, ensure_ascii=False)
hs, h0 = [x[2] for x in error_list], [x[3] for x in error_list]
print(f"done: {tgt} · {len(error_list)} matrices · relative weight error avg: stock conversion {100 * sum(h0) / len(h0):.2f}% → this {100 * sum(hs) / len(hs):.2f}% "
      f"(max {100 * max(hs):.2f}%, {max(error_list, key=lambda x: x[2])[0]}) · {time.time() - t0:.0f} s")
json.dump([{"module": m, "k": k, "error": h, "error_stock": h_} for m, k, h, h_ in error_list], open(os.path.join(tgt, "compression-report.json"), "w"), indent=0)
