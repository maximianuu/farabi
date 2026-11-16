"""Quantizes Gemma's official draft model (Gemma4Assistant, 0.94 GB bf16) with our own nvfp4 encoder (2026-11-11).

Why: the time of a draft step is almost entirely memory reads (each step reads the 262,144 × 1,024 vocabulary head + 4 layers ≈ 0.9 GB). In 4 bit ≈ 0.26 GB.
Measurement (long-form writing request, greedy): bf16 50.8 → quantized 54.5 tokens/s; the draft acceptance rate did not drop (draft error does not corrupt the output, it only affects acceptance).
No shift (k=0): no compensation needed; block scales are still chosen with least squares (tools/nvfp4.py).

Usage: python tools/gemma_draft.py <original draft folder> <output folder>
"""
import glob, json, os, shutil, sys, time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvfp4 as N

source, tgt = sys.argv[1:3]
os.makedirs(tgt, exist_ok=False)
t0 = time.time()
A = {}
for f in sorted(glob.glob(os.path.join(source, "*.safetensors"))):
    A.update(mx.load(f))
outputs, error_list = {}, []
for name, v in A.items():
    if name.endswith(".weight") and v.ndim == 2 and v.shape[1] % 16 == 0 and min(v.shape) >= 512:
        W = v.astype(mx.float32)
        w, s = N.compress(W, 0, chunk=8192)
        h = (mx.linalg.norm((N.dequant(w, s) - W).reshape(-1)) / mx.linalg.norm(W.reshape(-1))).item()
        outputs[name], outputs[name[: -len("weight")] + "scales"] = w, s
        error_list.append((name, h)); mx.eval(w, s); mx.clear_cache()
    else:
        outputs[name] = v
mx.save_safetensors(os.path.join(tgt, "model.safetensors"), outputs, metadata={"format": "mlx"})
for f in os.listdir(source):
    if not f.endswith(".safetensors") and os.path.isfile(os.path.join(source, f)):
        shutil.copy(os.path.join(source, f), os.path.join(tgt, f))
cfg = json.load(open(os.path.join(tgt, "config.json")))
cfg["quantization"] = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
cfg["farabi_compression"] = "nvfp4 · least-squares block scale, no shift (tools/gemma_draft.py)"
json.dump(cfg, open(os.path.join(tgt, "config.json"), "w"), indent=2, ensure_ascii=False)
print(f"done: {tgt} · {len(error_list)} matrices quantized · relative error avg {100 * sum(h for _, h in error_list) / len(error_list):.2f}% (max {100 * max(h for _, h in error_list):.2f}%, {max(error_list, key=lambda x: x[1])[0]}) · "
      f"{sum(v.nbytes for v in outputs.values()) / 1e9:.2f} GB · {time.time() - t0:.0f} s")
