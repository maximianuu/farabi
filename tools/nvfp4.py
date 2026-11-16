"""Better nvfp4 quantization (2026-11-10).

In the output of MLX's `mx.quantize(..., mode="nvfp4")` the block scales fall into the subnormal region of E4M3 (typical block scale for Qwen weights ≈ 0.005;
in that region the step is 0.00195 → effectively only 3 distinct scales in a layer). Here:
  1) the weight is scaled up by 2^k (k an integer → EXACT in floating point; the 2^-k compensation is folded by the caller into a norm weight etc.),
     so the scales land in the well-resolved region of E4M3 (step 12.5%);
  2) for every block of 16, the representable scale that minimizes the squared error is chosen (not the one closest to max/6).
The format is the same as MLX's (8 four-bit values per U32, U8 E4M3 scale); `mx.dequantize` / `quantized_matmul` decode it as is.
"""
import math

import mlx.core as mx

FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
THRESH = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def e4m3_table() -> mx.array:
    """code (0..126) → value; 127 is NaN, so it is not used."""
    v = [0.0] + [2.0 ** -9 * m for m in range(1, 8)] + [2.0 ** (e - 7) * (1 + m / 8) for e in range(1, 16) for m in range(8)]
    return mx.array(v[:127], dtype=mx.float32)


TABLE = e4m3_table()


def choose_shift(W: mx.array, margin: float = 1.0) -> int:
    """The largest k such that the largest block scale does not exceed the E4M3 upper limit (448)."""
    peak = mx.abs(W).max().item() / 6.0
    return int(math.floor(math.log2(448.0 * margin / max(peak, 1e-30))))


def compress(W: mx.array, k: int = 0, candidates=range(-9, 3), chunk: int = 2048):
    """W (out, in) float → (w U32 (out, in/8), scale U8 (out, in/16)). Dequantized ≈ W·2^k."""
    outp, inp = W.shape
    assert inp % 16 == 0
    ws, ss = [], []
    for begin in range(0, outp, chunk):
        B = (W[begin:begin + chunk].astype(mx.float32) * (2.0 ** k)).reshape(-1, inp // 16, 16)
        A = mx.abs(B)
        tgt = A.max(axis=-1) / 6.0
        # nearest code: normal-region formula (subnormals and zero are clipped)
        h = mx.maximum(tgt, 2.0 ** -9)
        e = mx.floor(mx.log2(h))
        m = mx.round((h / mx.power(2.0, e) - 1.0) * 8.0)
        code0 = mx.clip(((e + 7) * 8 + m), 1, 126).astype(mx.int32)
        code0 = mx.where(h < 2.0 ** -6, mx.clip(mx.round(h / 2.0 ** -9), 1, 8).astype(mx.int32), code0)
        best_err = best_code = None
        for d in candidates:
            code = mx.clip(code0 + d, 1, 126)
            s = TABLE[code][..., None]
            a = A / s
            n = sum((a > t).astype(mx.float32) for t in THRESH)
            q = (n * 0.5 + mx.maximum(n - 4, 0) * 0.5 + mx.maximum(n - 6, 0) * 1.0) * s      # 0..7 → FP4 value
            err = mx.square(A - q).sum(axis=-1)
            if best_err is None:
                best_err, best_code = err, code
            else:
                better = err < best_err
                best_err, best_code = mx.where(better, err, best_err), mx.where(better, code, best_code)
            mx.eval(best_err, best_code)
        s = TABLE[best_code][..., None]
        a = A / s
        n = sum((a > t).astype(mx.uint32) for t in THRESH)
        n = n + (B < 0).astype(mx.uint32) * 8
        n = n.reshape(-1, inp // 8, 8)
        w = sum(n[..., i] << (4 * i) for i in range(8)).astype(mx.uint32)
        mx.eval(w, best_code)
        ws.append(w); ss.append(best_code.astype(mx.uint8))
    return mx.concatenate(ws, axis=0), mx.concatenate(ss, axis=0)


def dequant(w: mx.array, s: mx.array) -> mx.array:
    return mx.dequantize(w, s, group_size=16, bits=4, mode="nvfp4").astype(mx.float32)


def orig_weights(orig_path: str) -> dict:
    """Original (HF) weights: lazy dict (name → mx.array), across all shards."""
    import glob, os
    d = {}
    for f in sorted(glob.glob(os.path.join(orig_path, "model-*.safetensors"))):
        d.update(mx.load(f))
    return d


def orig_name(module_name: str) -> str:
    """Module name inside the engine's language model → weight name in the original repository."""
    return "lm_head.weight" if module_name == "lm_head" else "model.language_model." + module_name[len("model."):] + ".weight"


def apply_in_memory(engine, orig_path: str, log_fn=print, candidates=range(-9, 3)):
    """MEASUREMENT: re-creates the engine's quantized layers in memory from the original weights with this quantizer.
    The compensation (2^-k) is applied at run time (`_global`); in the saved model it is folded into the norm weights (see tools/convert3.py)."""
    import time
    import mlx.nn as nn
    W = orig_weights(orig_path)
    t = time.time(); cnt = 0
    for name, m in engine.lm.named_modules():
        if not isinstance(m, nn.QuantizedLinear):
            continue
        w0 = W[orig_name(name)]
        k = choose_shift(w0)
        w, s = compress(w0, k, candidates=candidates)
        assert w.shape == m["weight"].shape and s.shape == m["scales"].shape, name
        m["weight"], m["scales"] = w, s
        m._global = 2.0 ** -k
        cnt += 1
        mx.clear_cache()
    log_fn(f"MEASUREMENT: {cnt} layers re-quantized (in memory) · {time.time() - t:.0f} s")
