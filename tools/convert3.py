"""Produces the engine's model with improved nvfp4 (2026-11-10). Source: Qwen's original weights + the stock MLX conversion (as the base).

What changes (see tools/nvfp4.py): every quantized matrix is scaled up by 2^k and the block scales are chosen with least squares; the 2^-k compensation
is folded EXACTLY (a power of two) into neighbouring weights, so it costs nothing at run time:
  linear-attention layer : in_proj_qkv, in_proj_z ← 2^k_g ; in_proj_a, in_proj_b (bf16) ← 2^k_g ; input_layernorm ← 2^-k_g
                           out_proj ← 2^k_c ; linear_attn.norm ← 2^-k_c
  full-attention layer   : q_proj, k_proj ← 2^k_g ; v_proj ← 2^k_v ; o_proj ← 2^k_o (k_v + k_o = k_g) ; input_layernorm ← 2^-k_g
  MLP                    : gate_proj ← 2^k_n ; up_proj ← 2^k_u ; down_proj ← 2^k_d (k_u + k_d = k_n) ; post_attention_layernorm ← 2^-k_n
  lm_head                : ← 2^k ; the compensation is a single multiply at run time (config.json → "farabi_lm_head_shift")
The model's function is mathematically identical; only the 4-bit rounding error shrinks (per layer ≈ 11% → 8.6%). Size and speed are the same.

Usage: python tools/convert3.py <original folder> <base MLX model (stock conversion)> <output folder>
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
file = {}                      # key → shard file name
A = {}                          # key → array (base; to be overwritten)
for f in chunks:
    for k, v in mx.load(f).items():
        A[k] = v; file[k] = os.path.basename(f)
LM = "language_model.model.layers."
error_list, lowest = [], []


def kmax(name):
    return N.choose_shift(W[name])


def write_out(mod_name, k):
    """module: 'layers.3.self_attn.q_proj' or 'lm_head'."""
    orig_key = "lm_head.weight" if mod_name == "lm_head" else f"model.language_model.{mod_name}.weight"
    hk = "language_model.lm_head" if mod_name == "lm_head" else f"language_model.model.{mod_name}"
    w0 = W[orig_key].astype(mx.float32)
    w, s = N.compress(w0, k)
    assert A[hk + ".weight"].shape == w.shape and A[hk + ".scales"].shape == s.shape, mod_name
    h = (mx.linalg.norm((N.dequant(w, s) * 2.0 ** -k - w0).reshape(-1)) / mx.linalg.norm(w0.reshape(-1))).item()
    under = (s < 8).astype(mx.float32).mean().item()
    A[hk + ".weight"], A[hk + ".scales"] = w, s
    mx.eval(w, s)
    error_list.append((mod_name, k, h, under))
    mx.clear_cache()


def mul_pow2(entry_key, us):
    v = A[entry_key]
    A[entry_key] = (v.astype(mx.float32) * (2.0 ** us)).astype(v.dtype)       # power of two: exact in bf16


def divide(total_tgt, a_cap, b_ust):
    """Two shifts such that a + b = total target; both are the same distance from their upper limit."""
    enabled = a_cap + b_ust - total_tgt
    assert enabled >= 0, (total_tgt, a_cap, b_ust)
    a = a_cap - enabled // 2
    return a, total_tgt - a


n = 1 + max(int(k[len(LM):].split(".")[0]) for k in A if k.startswith(LM))
for i in range(n):
    p = f"layers.{i}."
    if f"{LM}{i}.linear_attn.in_proj_qkv.weight" in A:
        kg = min(kmax(f"model.language_model.{p}linear_attn.in_proj_qkv.weight"), kmax(f"model.language_model.{p}linear_attn.in_proj_z.weight"))
        write_out(p + "linear_attn.in_proj_qkv", kg); write_out(p + "linear_attn.in_proj_z", kg)
        mul_pow2(f"{LM}{i}.linear_attn.in_proj_a.weight", kg); mul_pow2(f"{LM}{i}.linear_attn.in_proj_b.weight", kg)
        mul_pow2(f"{LM}{i}.input_layernorm.weight", -kg)
        kc = kmax(f"model.language_model.{p}linear_attn.out_proj.weight")
        write_out(p + "linear_attn.out_proj", kc); mul_pow2(f"{LM}{i}.linear_attn.norm.weight", -kc)
    else:
        kg = min(kmax(f"model.language_model.{p}self_attn.q_proj.weight"), kmax(f"model.language_model.{p}self_attn.k_proj.weight"))
        write_out(p + "self_attn.q_proj", kg); write_out(p + "self_attn.k_proj", kg)
        kv, ko = divide(kg, kmax(f"model.language_model.{p}self_attn.v_proj.weight"), kmax(f"model.language_model.{p}self_attn.o_proj.weight"))
        write_out(p + "self_attn.v_proj", kv); write_out(p + "self_attn.o_proj", ko)
        mul_pow2(f"{LM}{i}.input_layernorm.weight", -kg)
    kn = kmax(f"model.language_model.{p}mlp.gate_proj.weight")
    write_out(p + "mlp.gate_proj", kn)
    ku, kd = divide(kn, kmax(f"model.language_model.{p}mlp.up_proj.weight"), kmax(f"model.language_model.{p}mlp.down_proj.weight"))
    write_out(p + "mlp.up_proj", ku); write_out(p + "mlp.down_proj", kd)
    mul_pow2(f"{LM}{i}.post_attention_layernorm.weight", -kn)
    if i % 8 == 7:
        latest = error_list[-11:]
        print(f"layer {i + 1}/{n} · {time.time() - t0:.0f} s · relative error of the last layers {100 * sum(h for _, _, h, _ in latest) / len(latest):.2f}%", flush=True)
klm = kmax("lm_head.weight")
write_out("lm_head", klm)

for f in chunks:
    name = os.path.basename(f)
    mx.save_safetensors(os.path.join(tgt, name), {k: v for k, v in A.items() if file[k] == name}, metadata={"format": "mlx"})
    print("written:", name, flush=True)
for f in os.listdir(base_dir):
    if not f.endswith(".safetensors") and os.path.isfile(os.path.join(base_dir, f)):
        shutil.copy(os.path.join(base_dir, f), os.path.join(tgt, f))
cfg = json.load(open(os.path.join(tgt, "config.json")))
cfg["farabi_lm_head_shift"] = klm
cfg["farabi_compression"] = "nvfp4 · 2^k shift + least-squares block scale (tools/convert3.py)"
json.dump(cfg, open(os.path.join(tgt, "config.json"), "w"), indent=2, ensure_ascii=False)
hs = [h for _, _, h, _ in error_list]
print(f"done: {tgt} · {len(error_list)} matrices · relative error avg {100 * sum(hs) / len(hs):.2f}% max {100 * max(hs):.2f}% "
      f"({max(error_list, key=lambda x: x[2])[0]}) · max fraction of blocks with subnormal scales {100 * max(a for *_, a in error_list):.2f}% · lm_head k={klm} · {time.time() - t0:.0f} s")
json.dump([{"module": m, "k": k, "error": h, "subnormal": a} for m, k, h, a in error_list], open(os.path.join(tgt, "compression-report.json"), "w"), indent=0)
