"""Error-compensated nvfp4 quantization (GPTQ, Frantar et al. 2022) — on MLX.

While rounding a matrix column by column, the rounding error is distributed to the columns that have not been rounded yet; the distribution uses the second moment
of the layer's real inputs (H = XᵀX), i.e. the goal is to reduce the OUTPUT error (‖WX − ŴX‖²), not the weight error.
Block scales (chunks of 16 columns × row) are chosen with least squares among the representable E4M3 scales, according to the compensated values.
The format is the same as MLX nvfp4 (tools/nvfp4.py): 8 four-bit values per U32 + a U8 scale; the weight is stored scaled up by 2^k.
"""
import mlx.core as mx

from nvfp4 import THRESH, TABLE


def upper_chol_inv(H: mx.array) -> mx.array:
    """Upper-triangular U with H⁻¹ = UᵀU. H = M Mᵀ (M upper-triangular; reverse-order Cholesky) → U = M⁻¹."""
    L = mx.linalg.cholesky(H[::-1, ::-1], stream=mx.cpu)
    return mx.linalg.tri_inv(L[::-1, ::-1], upper=True, stream=mx.cpu)


def _choose_scale(G: mx.array, candidates) -> mx.array:
    """G (rows, 16) → least-squares E4M3 code per row."""
    A = mx.abs(G)
    h = mx.maximum(A.max(axis=-1) / 6.0, 2.0 ** -9)
    e = mx.floor(mx.log2(h))
    m = mx.round((h / mx.power(2.0, e) - 1.0) * 8.0)
    code0 = mx.clip((e + 7) * 8 + m, 1, 126).astype(mx.int32)
    code0 = mx.where(h < 2.0 ** -6, mx.clip(mx.round(h / 2.0 ** -9), 1, 8).astype(mx.int32), code0)
    best_err = best_code = None
    for d in candidates:
        code = mx.clip(code0 + d, 1, 126)
        s = TABLE[code][:, None]
        a = A / s
        n = sum((a > t).astype(mx.float32) for t in THRESH)
        q = (n * 0.5 + mx.maximum(n - 4, 0) * 0.5 + mx.maximum(n - 6, 0)) * s
        err = mx.square(A - q).sum(axis=-1)
        if best_err is None:
            best_err, best_code = err, code
        else:
            better = err < best_err
            best_err, best_code = mx.where(better, err, best_err), mx.where(better, code, best_code)
    return best_code


def gptq(W: mx.array, H: mx.array, k: int, block: int = 128, damping: float = 0.01, candidates=range(-6, 3)):
    """W (rows, columns) original weight, H (columns, columns) input second moment → (w U32, scale U8). Dequantized ≈ W·2^k."""
    row, col = W.shape
    assert col % 16 == 0 and block % 16 == 0
    Ws = W.astype(mx.float32) * (2.0 ** k)
    H = H.astype(mx.float32)
    d = mx.diagonal(H)
    dead = d <= 0
    H = H + mx.diag(mx.where(dead, 1.0, 0.0)) + mx.eye(col) * (damping * d.mean())
    Ws = mx.where(dead[None, :], 0.0, Ws)
    U = upper_chol_inv(H)
    mx.eval(U, Ws)
    codes, quads = [], []
    for i1 in range(0, col, block):
        i2 = min(i1 + block, col)
        c = i2 - i1
        W1, U1 = Ws[:, i1:i2], U[i1:i2, i1:i2]
        error_list, nb = [], []
        s = None
        for i in range(c):
            w = W1[:, i]
            if i % 16 == 0:
                code = _choose_scale(W1[:, i:i + 16], candidates)
                s = TABLE[code]
                codes.append(code)
            a = mx.abs(w) / s
            n = sum((a > t).astype(mx.float32) for t in THRESH)
            q = (n * 0.5 + mx.maximum(n - 4, 0) * 0.5 + mx.maximum(n - 6, 0)) * s * mx.sign(w)
            e = (w - q) / U1[i, i]
            if i + 1 < c:
                W1 = mx.concatenate([W1[:, : i + 1], W1[:, i + 1:] - e[:, None] * U1[i, i + 1:][None, :]], axis=1)
            error_list.append(e)
            nb.append(n.astype(mx.uint32) + (w < 0).astype(mx.uint32) * 8)
            if i % 16 == 15:
                mx.eval(W1, error_list[-16:], nb[-16:], code)
        quads.append(mx.stack(nb, axis=1))
        if i2 < col:
            Ws = mx.concatenate([Ws[:, :i2], Ws[:, i2:] - mx.stack(error_list, axis=1) @ U[i1:i2, i2:]], axis=1)
        mx.eval(Ws, quads[-1])
    n = mx.concatenate(quads, axis=1).reshape(row, col // 8, 8)
    w = sum(n[..., i] << (4 * i) for i in range(8)).astype(mx.uint32)
    s = mx.stack(codes, axis=1).astype(mx.uint8)
    mx.eval(w, s)
    return w, s


def output_err(W: mx.array, Wq: mx.array, H: mx.array) -> float:
    """‖(W − Ŵ)X‖² / ‖WX‖² (via H = XᵀX)."""
    D = (W - Wq).astype(mx.float32)
    return ((D @ H) * D).sum().item() / max(((W.astype(mx.float32) @ H) * W.astype(mx.float32)).sum().item(), 1e-30)
