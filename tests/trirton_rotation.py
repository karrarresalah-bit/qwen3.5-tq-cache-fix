"""
Triton Drama Level 1 — The Rotation Kernel 🌪️
===============================================
We implement the TurboQuant rotation step as a Triton kernel:

    output[i] = normalize(input[i]) @ R.T

Where:
    input  : [N, 256]  — N KV vectors, each 256-dimensional
    R      : [256, 256] — our fixed rotation matrix
    output : [N, 256]  — rotated unit vectors

This is a FUSED kernel: normalize + matmul in ONE GPU trip!

The new concept: TILING
    The rotation matrix is 256KB — too big for SRAM.
    We cut it into tiles and accumulate the result slice by slice.
    Like eating a pizza one slice at a time! 🍕

Run:
    python triton_rotation.py
"""

import torch
import triton
import triton.language as tl
import time

# Our exact KV dimensions from step 1
HEAD_DIM   = 256
TILE_SIZE  = 64    # how big each pizza slice is (must divide HEAD_DIM evenly)
            # 256 / 64 = 4 slices — fits perfectly!


# ═════════════════════════════════════════════════════════════════════════════
# The Kernel
# ═════════════════════════════════════════════════════════════════════════════

@triton.jit
def normalize_and_rotate_kernel(
    # Input: KV vectors [N, HEAD_DIM]
    input_ptr,
    input_stride_row,   # how many elements to skip per row (= HEAD_DIM for contiguous)

    # Rotation matrix [HEAD_DIM, HEAD_DIM]
    # We apply R.T, so we read R in transposed order
    rot_ptr,
    rot_stride_row,     # stride for rotation matrix rows

    # Output: rotated unit vectors [N, HEAD_DIM]
    output_ptr,
    output_stride_row,

    # Also output the per-vector norm (we need it for decompression later!)
    norms_ptr,          # [N] — one norm per vector

    # Dimensions
    N,                  # number of vectors
    HEAD_DIM: tl.constexpr,   # vector dimension (256)
    TILE_SIZE: tl.constexpr,  # pizza slice size (64)
):
    """
    Each chef handles ONE input vector (one row).
    Chef i:
        1. Loads row i from input            [HEAD_DIM floats]
        2. Computes its norm, normalizes it
        3. Multiplies by R.T using tiling    [HEAD_DIM output floats]
        4. Stores the result + norm
    """

    # Which vector am I? (which row)
    row_id = tl.program_id(axis=0)
    if row_id >= N:
        return

    # ── STEP 1: Load my input vector ─────────────────────────────────────────
    # My row starts at: input_ptr + row_id * input_stride_row
    row_ptr = input_ptr + row_id * input_stride_row
    col_offsets = tl.arange(0, HEAD_DIM)   # [0, 1, 2, ..., 255]

    # Load the full 256-element vector into registers
    x = tl.load(row_ptr + col_offsets).to(tl.float32)   # shape: [HEAD_DIM]

    # ── STEP 2: Normalize to unit length ─────────────────────────────────────
    # norm = sqrt(sum(x^2))
    norm = tl.sqrt(tl.sum(x * x))
    norm = tl.maximum(norm, 1e-8)   # safety: avoid division by zero

    # Save norm for the Python side (needed for decompression)
    tl.store(norms_ptr + row_id, norm)

    # Normalize
    x_unit = x / norm   # now x_unit has length 1.0

    # ── STEP 3: Multiply by R.T using TILING ─────────────────────────────────
    #
    # We want: output = x_unit @ R.T
    # Which means: output[j] = sum_k( x_unit[k] * R[j, k] )
    #
    # TILING PLAN:
    # Instead of loading all of R at once (too big!),
    # we process TILE_SIZE columns of R at a time:
    #
    #   output[j] = sum over tiles of:
    #                   x_unit[tile_start:tile_end] · R[j, tile_start:tile_end]
    #
    # We accumulate into `acc` as we go through tiles.
    #
    # ANALOGY: Computing a dot product by reading the book in chapters 📚
    #   Chapter 1: read pages 0-63,   add to running total
    #   Chapter 2: read pages 64-127, add to running total
    #   Chapter 3: read pages 128-191, add to running total
    #   Chapter 4: read pages 192-255, add to running total
    #   Done! Running total = final dot product

    # Accumulator: one value per output dimension
    # Shape: [HEAD_DIM] — we'll compute all output dims simultaneously!
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Output dimension indices: [0, 1, 2, ..., HEAD_DIM-1]
    out_dims = tl.arange(0, HEAD_DIM)

    # Loop over tiles of the input/rotation
    for tile_start in range(0, HEAD_DIM, TILE_SIZE):
        # Which input elements are in this tile?
        tile_offsets = tile_start + tl.arange(0, TILE_SIZE)  # [tile_start..tile_start+63]

        # Load this tile of x_unit: shape [TILE_SIZE]
        x_tile = tl.load(row_ptr + tile_offsets).to(tl.float32) / norm
        # (we reload from memory and renormalize — avoids storing x_unit separately)

        # Load the corresponding tile of R
        # For output dim j and input tile element k:
        # R[j, k] is at: rot_ptr + j * rot_stride_row + k
        #
        # We want a [HEAD_DIM, TILE_SIZE] block of R
        # out_dims[:, None] = [[0], [1], ..., [255]]  shape [HEAD_DIM, 1]
        # tile_offsets[None, :] = [[tile_start, ..., tile_start+63]] shape [1, TILE_SIZE]
        # Together: [HEAD_DIM, TILE_SIZE] block!
        r_block_ptr = rot_ptr + out_dims[:, None] * rot_stride_row + tile_offsets[None, :]
        r_block = tl.load(r_block_ptr).to(tl.float32)
        # r_block shape: [HEAD_DIM, TILE_SIZE]

        # Accumulate: for each output dim j, add dot product with this tile
        # x_tile shape:  [TILE_SIZE]
        # r_block shape: [HEAD_DIM, TILE_SIZE]
        # sum over TILE_SIZE dim → [HEAD_DIM]
        acc += tl.sum(r_block * x_tile[None, :], axis=1)

    # ── STEP 4: Store the result ──────────────────────────────────────────────
    out_ptr = output_ptr + row_id * output_stride_row
    tl.store(out_ptr + out_dims, acc)


# ═════════════════════════════════════════════════════════════════════════════
# Python launcher
# ═════════════════════════════════════════════════════════════════════════════

def normalize_and_rotate(
    x: torch.Tensor,        # [N, 256] float16 or float32
    R: torch.Tensor,        # [256, 256] rotation matrix
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused normalize + rotate.
    Returns:
        rotated : [N, 256] float32 — unit vectors rotated by R
        norms   : [N]      float32 — original vector norms (for decompression)
    """
    assert x.ndim == 2 and x.shape[1] == HEAD_DIM
    assert R.shape == (HEAD_DIM, HEAD_DIM)
    assert x.is_cuda and R.is_cuda

    # Make contiguous — Triton needs contiguous memory!
    x = x.contiguous().float()
    R = R.contiguous().float()

    N = x.shape[0]
    rotated = torch.empty(N, HEAD_DIM, device=x.device, dtype=torch.float32)
    norms   = torch.empty(N,           device=x.device, dtype=torch.float32)

    # One chef per row
    grid = (N,)

    normalize_and_rotate_kernel[grid](
        x,       x.stride(0),
        R,       R.stride(0),
        rotated, rotated.stride(0),
        norms,
        N,
        HEAD_DIM=HEAD_DIM,
        TILE_SIZE=TILE_SIZE,
    )

    return rotated, norms


# ═════════════════════════════════════════════════════════════════════════════
# PyTorch reference (what we're replacing)
# ═════════════════════════════════════════════════════════════════════════════

def normalize_and_rotate_pytorch(x, R):
    x_f    = x.float()
    norms  = x_f.norm(dim=1, keepdim=True).clamp(min=1e-8)
    x_unit = x_f / norms
    rotated = x_unit @ R.T
    return rotated, norms.squeeze(1)


# ═════════════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════════════

def sep(title=""):
    w = 62
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("\n" + "─" * w)


def main():
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU!")
        return

    print(f"\n  GPU      : {torch.cuda.get_device_name(0)}")
    print(f"  HEAD_DIM : {HEAD_DIM}")
    print(f"  TILE_SIZE: {TILE_SIZE}  ({HEAD_DIM // TILE_SIZE} tiles per vector)")

    # Build our rotation matrix
    torch.manual_seed(42)
    Q, _ = torch.linalg.qr(torch.randn(HEAD_DIM, HEAD_DIM))
    R = Q.cuda()

    # ── Test 1: correctness on small input ───────────────────────────────────
    sep("Test 1 — Correctness check")
    torch.manual_seed(0)
    x = torch.randn(128, HEAD_DIM, device="cuda") * 0.1
    # Add outliers (like real KV vectors)
    x[5,  :8] *= 15.0
    x[42, :4] *= 20.0

    triton_out,  triton_norms  = normalize_and_rotate(x, R)
    pytorch_out, pytorch_norms = normalize_and_rotate_pytorch(x, R)

    diff_rot   = (triton_out   - pytorch_out).abs().max().item()
    diff_norms = (triton_norms - pytorch_norms).abs().max().item()

    status_rot   = "✅ PASS" if diff_rot   < 1e-3 else "❌ FAIL"
    status_norms = "✅ PASS" if diff_norms < 1e-3 else "❌ FAIL"
    print(f"  {status_rot}   rotation  (max diff = {diff_rot:.2e})")
    print(f"  {status_norms}   norms     (max diff = {diff_norms:.2e})")

    # ── Test 2: verify unit length after rotation ─────────────────────────────
    sep("Test 2 — Are output vectors unit length?")
    row_norms = triton_out.norm(dim=1)
    print(f"  min norm: {row_norms.min():.6f}  (should be ~1.0)")
    print(f"  max norm: {row_norms.max():.6f}  (should be ~1.0)")
    ok = (row_norms - 1.0).abs().max().item() < 1e-3
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  all rows are unit vectors")

    # ── Test 3: realistic KV cache shape ─────────────────────────────────────
    sep("Test 3 — Realistic KV cache shape")
    # From step 1: [4 heads, seq_len, 256] flattened to [4*seq_len, 256]
    seq_len  = 1024
    n_heads  = 4
    kv = torch.randn(n_heads * seq_len, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1
    print(f"  Input shape: {list(kv.shape)}  (dtype: {kv.dtype})")

    out, norms = normalize_and_rotate(kv, R)
    print(f"  Output shape: {list(out.shape)}  (dtype: {out.dtype})")
    print(f"  Norms shape:  {list(norms.shape)}")
    print(f"  ✅ Processed {n_heads * seq_len} vectors in one kernel launch!")

    # ── Speed comparison ──────────────────────────────────────────────────────
    sep("Speed: Triton fused vs PyTorch two-step")

    x_bench = torch.randn(4096, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1

    # Warmup
    for _ in range(20):
        normalize_and_rotate(x_bench, R)
        normalize_and_rotate_pytorch(x_bench, R)
    torch.cuda.synchronize()

    # Triton
    t0 = time.perf_counter()
    for _ in range(1000):
        normalize_and_rotate(x_bench, R)
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) / 1000 * 1e6

    # PyTorch
    t0 = time.perf_counter()
    for _ in range(1000):
        normalize_and_rotate_pytorch(x_bench, R)
    torch.cuda.synchronize()
    t_pytorch = (time.perf_counter() - t0) / 1000 * 1e6

    speedup = t_pytorch / t_triton
    print(f"  Triton fused  : {t_triton:.1f} µs")
    print(f"  PyTorch 2-step: {t_pytorch:.1f} µs")
    print(f"  Speedup       : {speedup:.2f}x  {'🔥 Triton wins!' if speedup > 1.0 else '🤔 Close!'}")
    print(f"  (fusion saves one full read of [{4096}×{HEAD_DIM}] from HBM)")

    sep("What Drama 1 taught us!")
    print("""
  NEW concepts mastered:
  ─────────────────────
  tl.zeros([N], dtype)         → accumulator for tiling loop
  ptr + row[:, None] + col[None, :] → 2D block addressing  ← key pattern!
  tl.sum(block, axis=1)        → reduce along tile dimension
  for tile in range(0, D, T)   → the tiling loop

  The pizza slice pattern:
    acc = zeros
    for each slice of rotation matrix:
        acc += x_slice · R_slice
    result = acc   ✅

  Next: Drama Level 2 — add quantization to this kernel!
  We'll find the nearest codebook entry for each of the 256
  rotated values, ALL inside the same kernel. 🎯
    """)


if __name__ == "__main__":
    main()