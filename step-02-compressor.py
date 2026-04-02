"""
Step 2 — TurboQuant Standalone Compressor (FIXED)
==================================================
Fixes from v1:
  - Per-vector normalization (mu/std) so codebook actually matches data
  - Store mu+std alongside scale for proper decompression
  - Consistent float32 throughout the math
  - Bit-packing for honest compression ratios
"""

import torch
import time

HEAD_DIM     = 256
NUM_KV_HEADS = 4
SEQ_LEN      = 512


# ── Rotation matrix ───────────────────────────────────────────────────────────
# Same as before — build once, reuse forever
def make_rotation_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(dim, dim))
    return Q


# ── Codebook ──────────────────────────────────────────────────────────────────
# Evenly spaced values from -3 to +3 (covers 99.7% of a Gaussian)
def make_codebook(bits: int) -> torch.Tensor:
    return torch.linspace(-3.0, 3.0, 2 ** bits)


# ── Compress one vector ───────────────────────────────────────────────────────
# THE KEY FIX: after rotating, normalize each vector to zero-mean unit-std
# BEFORE quantizing. Then store mu and std so we can undo it later.
#
# Why did v1 fail? The codebook lives in [-3, 3] but after rotation the values
# might live in [-0.05, 0.05] — they never matched! Like trying to buy shoes
# in EU sizes when all you have is US sizes and no conversion. 👟
#
# Now we convert: (value - mu) / std  →  lives in roughly [-3, 3]  →  quantize!
def compress_vector(x, rotation, codebook):
    x = x.float()

    # Save length, normalize to unit vector
    scale = x.norm().item()
    if scale < 1e-8:
        return torch.zeros(x.shape[0], dtype=torch.long), scale, 0.0, 1.0

    x_unit    = x / scale
    x_rotated = rotation @ x_unit          # spread energy evenly

    # NEW: normalize to match codebook range [-3, 3]
    mu  = x_rotated.mean().item()
    std = x_rotated.std().item() + 1e-8    # +epsilon to avoid div by zero
    x_normalized = (x_rotated - mu) / std  # now roughly Gaussian with mean=0, std=1

    # Quantize: find nearest codebook entry for each of the 256 numbers
    distances = (x_normalized.unsqueeze(1) - codebook.unsqueeze(0)).abs()
    indices   = distances.argmin(dim=1)    # shape [256], values 0..n_levels-1

    return indices, scale, mu, std


# ── Decompress one vector ─────────────────────────────────────────────────────
def decompress_vector(indices, scale, mu, std, rotation, codebook):
    # Look up codebook → undo normalization → rotate backwards → restore scale
    x_norm_approx    = codebook[indices].float()
    x_rotated_approx = x_norm_approx * std + mu   # undo normalization
    x_unit_approx    = rotation.T @ x_rotated_approx  # rotate backwards
    return (x_unit_approx * scale).half()


# ── Compress full KV tensor ───────────────────────────────────────────────────
# Shape in: [4 heads, seq_len, 256]
# Stores: indices [4, seq_len, 256], scales [4, seq_len], mu [4, seq_len], std [4, seq_len]
def compress_kv(kv, rotation, codebook):
    n_heads, seq_len, head_dim = kv.shape
    all_indices = torch.zeros(n_heads, seq_len, head_dim, dtype=torch.long)
    all_scales  = torch.zeros(n_heads, seq_len)
    all_mu      = torch.zeros(n_heads, seq_len)
    all_std     = torch.zeros(n_heads, seq_len)

    for h in range(n_heads):
        for t in range(seq_len):
            idx, scale, mu, std = compress_vector(kv[h, t], rotation, codebook)
            all_indices[h, t] = idx
            all_scales[h, t]  = scale
            all_mu[h, t]      = mu
            all_std[h, t]     = std

    return all_indices, all_scales, all_mu, all_std


# ── Decompress full KV tensor ─────────────────────────────────────────────────
def decompress_kv(all_indices, all_scales, all_mu, all_std, rotation, codebook):
    n_heads, seq_len, _ = all_indices.shape
    out = torch.zeros(n_heads, seq_len, HEAD_DIM, dtype=torch.float16)

    for h in range(n_heads):
        for t in range(seq_len):
            out[h, t] = decompress_vector(
                all_indices[h, t],
                all_scales[h, t].item(),
                all_mu[h, t].item(),
                all_std[h, t].item(),
                rotation,
                codebook,
            )
    return out


# ── Memory calculation (honest, with bit-packing) ────────────────────────────
# Bit-packing: for N-bit quantization, pack multiple indices into one byte
# e.g. 4-bit → 2 indices per byte → half the storage of naive int8
def compressed_bytes(indices, scales, mu, std, bits):
    n_indices   = indices.numel()
    packed_bytes = (n_indices * bits + 7) // 8   # ceiling division for bit-packing
    meta_bytes   = (scales.numel() + mu.numel() + std.numel()) * 4  # float32 each
    return packed_bytes + meta_bytes


# ── Metrics ───────────────────────────────────────────────────────────────────
def cosine_sim(a, b):
    a_f = a.reshape(-1, a.shape[-1]).float()
    b_f = b.reshape(-1, b.shape[-1]).float()
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=1).mean().item()


def sep(title=""):
    w = 62
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("\n" + "─" * w)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sep("TurboQuant Compressor v2 — Fixed!")
    print(f"  HEAD_DIM={HEAD_DIM}, NUM_KV_HEADS={NUM_KV_HEADS}, SEQ_LEN={SEQ_LEN}")

    # Build rotation matrix and verify it
    sep("Rotation matrix")
    R = make_rotation_matrix(HEAD_DIM)
    err = (R.T @ R - torch.eye(HEAD_DIM)).abs().max().item()
    print(f"  Shape            : {list(R.shape)}")
    print(f"  Orthogonality err: {err:.2e}  ✅" if err < 1e-5 else f"  {err:.2e}  ❌")

    # Build test KV tensor with outliers (realistic)
    sep("Test KV tensor")
    torch.manual_seed(0)
    kv = torch.randn(NUM_KV_HEADS, SEQ_LEN, HEAD_DIM) * 0.1
    kv[0, 10, :8] *= 15.0    # outlier in head 0 token 10
    kv[2, 77, :4] *= 20.0    # outlier in head 2 token 77
    kv = kv.half()            # float16, like a real KV cache
    print(f"  Shape  : {list(kv.shape)}")
    print(f"  Range  : [{kv.float().min():.3f}, {kv.float().max():.3f}]")
    print(f"  Std dev: {kv.float().std():.4f}")
    orig_kb = kv.numel() * 2 / 1024
    print(f"  Size   : {orig_kb:.1f} KB (fp16)")

    # ── Test each bit width ───────────────────────────────────────────────────
    sep("Compression tests (fixed)")
    print(f"\n  {'bits':>6} {'cosine':>8} {'MSE':>10} {'comp KB':>9} {'ratio':>7} {'verdict'}")
    print("  " + "─" * 58)

    for bits in [4, 3, 2]:
        cb = make_codebook(bits)

        t0 = time.perf_counter()
        indices, scales, mus, stds = compress_kv(kv.float(), R, cb)
        t_compress = time.perf_counter() - t0

        kv_approx = decompress_kv(indices, scales, mus, stds, R, cb)

        cos  = cosine_sim(kv, kv_approx)
        mse  = (kv.float() - kv_approx.float()).pow(2).mean().item()
        comp = compressed_bytes(indices, scales, mus, stds, bits) / 1024
        ratio = orig_kb / comp

        verdict = ("✅ Excellent" if cos > 0.99 else
                   "✅ Great"     if cos > 0.97 else
                   "🟡 Good"      if cos > 0.93 else
                   "⚠️  Degraded")

        print(f"  {bits:>6} {cos:>8.4f} {mse:>10.6f} {comp:>9.1f} {ratio:>7.2f}x  {verdict}")

    # ── Scaling test ──────────────────────────────────────────────────────────
    sep("Scaling — quality vs sequence length (4-bit)")
    cb4 = make_codebook(4)
    print(f"\n  {'seq_len':>10} {'cosine':>8} {'orig KB':>9} {'comp KB':>9} {'ratio':>7}")
    print("  " + "─" * 48)
    for slen in [64, 128, 256, 512, 1024, 2048]:
        kv_t = (torch.randn(NUM_KV_HEADS, slen, HEAD_DIM) * 0.1).half()
        idx, sc, mu, std = compress_kv(kv_t.float(), R, cb4)
        kv_a = decompress_kv(idx, sc, mu, std, R, cb4)
        cos  = cosine_sim(kv_t, kv_a)
        o_kb = kv_t.numel() * 2 / 1024
        c_kb = compressed_bytes(idx, sc, mu, std, 4) / 1024
        print(f"  {slen:>10} {cos:>8.4f} {o_kb:>9.1f} {c_kb:>9.1f} {o_kb/c_kb:>7.2f}x")

    sep("Done!")
    print()
    print("  ✅ cosine > 0.99 for 4-bit  → ready for step 3!")
    print("  🔥 Next: build HybridTurboQuantCache and plug into Qwen3.5\n")


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# Results on my machine (RTX 5070):
# -----------------------------------------------------------------------------
# ❯ python .\step-02-compressor.py

# ───────────── TurboQuant Compressor v2 — Fixed! ──────────────
#   HEAD_DIM=256, NUM_KV_HEADS=4, SEQ_LEN=512

# ────────────────────── Rotation matrix ───────────────────────
#   Shape            : [256, 256]
#   Orthogonality err: 7.15e-07  ✅

# ─────────────────────── Test KV tensor ───────────────────────
#   Shape  : [4, 512, 256]
#   Range  : [-2.473, 2.209]
#   Std dev: 0.1003
#   Size   : 1024.0 KB (fp16)

# ───────────────── Compression tests (fixed) ──────────────────

#     bits   cosine        MSE   comp KB   ratio verdict
#   ──────────────────────────────────────────────────────────
#        4   0.9932   0.000138     280.0    3.66x  ✅ Excellent
#        3   0.9706   0.000618     216.0    4.74x  ✅ Great
#        2   0.8685   0.003375     152.0    6.74x  ⚠️  Degraded

# ──────── Scaling — quality vs sequence length (4-bit) ────────

#      seq_len   cosine   orig KB   comp KB   ratio
#   ────────────────────────────────────────────────
#           64   0.9933     128.0      35.0    3.66x
#          128   0.9933     256.0      70.0    3.66x
#          256   0.9932     512.0     140.0    3.66x
#          512   0.9932    1024.0     280.0    3.66x
#         1024   0.9932    2048.0     560.0    3.66x
#         2048   0.9932    4096.0    1120.0    3.66x

# ─────────────────────────── Done! ────────────────────────────

#   ✅ cosine > 0.99 for 4-bit  → ready for step 3!
#   🔥 Next: build HybridTurboQuantCache and plug into Qwen3.5