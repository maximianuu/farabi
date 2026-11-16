"""Quantizes Gemma's embedding table (also the output head) with our own nvfp4 encoder (2026-11-11).

Why: in the previous conversion the table stays bf16 (2.8 GB); at every generation round the output head reads this table in full (memory bandwidth → speed). In 4 bit it is 0.83 GB.
It was not proven that the stock conversion damages the table (the problem was in the body), but our own encoder (2^k shift + least-squares block scale) is more faithful.
Compensation (the table is stored scaled up by 2^k): at the input embed_scale ← embed_scale · 2^-k, at the output logit · 2^-k (BEFORE the soft cap) — engine/gemma.py applies this
(config.json → "farabi_embed_shift"). The stock library does not know this field: this model is used only with the engine.

Usage: python tools/gemma_embed.py <previous conversion folder> <output folder>
"""
import glob, json, os, shutil, sys, time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvfp4 as N

source, tgt = sys.argv[1:3]
os.makedirs(tgt, exist_ok=False)
t0 = time.time()
NAME = "language_model.model.embed_tokens"
k = None
for f in sorted(glob.glob(os.path.join(source, "model-*.safetensors"))):
    A = dict(mx.load(f))
    if NAME + ".weight" in A and NAME + ".scales" not in A:
        W = A[NAME + ".weight"].astype(mx.float32)
        k = N.choose_shift(W)
        w, s = N.compress(W, k, chunk=8192)
        err = (mx.linalg.norm((N.dequant(w, s) * 2.0 ** -k - W).reshape(-1)) / mx.linalg.norm(W.reshape(-1))).item()
        h0w, h0s, *_ = mx.quantize(W, group_size=16, bits=4, mode="nvfp4")
        err0 = (mx.linalg.norm((mx.dequantize(h0w, h0s, group_size=16, bits=4, mode="nvfp4").astype(mx.float32) - W).reshape(-1)) / mx.linalg.norm(W.reshape(-1))).item()
        A[NAME + ".weight"], A[NAME + ".scales"] = w, s
        print(f"embedding table {W.shape}: k={k} · relative error: stock encoder {100 * err0:.2f}% → this {100 * err:.2f}% · {time.time() - t0:.0f} s", flush=True)
        del W
    mx.save_safetensors(os.path.join(tgt, os.path.basename(f)), A, metadata={"format": "mlx"})
    print("written:", os.path.basename(f), flush=True)
    del A; mx.clear_cache()
assert k is not None, "embedding table not found or already quantized"
for f in os.listdir(source):
    if not f.endswith(".safetensors") and os.path.isfile(os.path.join(source, f)):
        shutil.copy(os.path.join(source, f), os.path.join(tgt, f))
cfg = json.load(open(os.path.join(tgt, "config.json")))
cfg["farabi_embed_shift"] = k
cfg["farabi_compression"] = cfg.get("farabi_compression", "") + " + embedding table nvfp4 (tools/gemma_embed.py)"
json.dump(cfg, open(os.path.join(tgt, "config.json"), "w"), indent=2, ensure_ascii=False)
print(f"done: {tgt} · {time.time() - t0:.0f} s")
