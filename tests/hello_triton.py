"""
Hello Triton! 🍳 First Steps in the GPU Kitchen
=================================================
We start with the SIMPLEST possible Triton kernel:
    multiply every element of a vector by 2.0

That's it. No rotation, no quantization — just:
    output[i] = input[i] * 2.0

But by doing this tiny thing, you'll understand:
    ✅ What a kernel function IS
    ✅ What pointers are and how to use them
    ✅ What BLOCK_SIZE means (our "chunk from freezer")
    ✅ How to launch a kernel
    ✅ How to verify it worked

Install triton first:
    pip install triton

Then run:
    python hello_triton.py
"""

import torch
import triton
import triton.language as tl   # tl = "triton language", the GPU-side functions


# ═════════════════════════════════════════════════════════════════════════════
# LESSON 1: What is a kernel?
#
# A kernel is a function that runs ON THE GPU, not on Python.
# Every "thread" on the GPU runs this function simultaneously.
# We don't write loops — we write what ONE thread does,
# and the GPU runs millions of them at once!
#
# The @triton.jit decorator means:
#   "compile this function for the GPU, not for Python"
# ═════════════════════════════════════════════════════════════════════════════

@triton.jit
def multiply_by_two_kernel(
    input_ptr,      # 📍 pointer: "where in GPU memory does my input data start?"
    output_ptr,     # 📍 pointer: "where should I write my output?"
    n_elements,     # 📊 plain number: how many elements are in the vector?
    BLOCK_SIZE: tl.constexpr,  # 🍱 how big is our "counter chunk"? (must be power of 2)
):
    """
    This function runs on the GPU.
    Each "program" (group of threads) handles one BLOCK_SIZE chunk of the data.

    ANALOGY: Imagine 1000 chefs all working simultaneously.
    Chef 0 handles elements [0..BLOCK_SIZE-1]
    Chef 1 handles elements [BLOCK_SIZE..2*BLOCK_SIZE-1]
    Chef 2 handles elements [2*BLOCK_SIZE..3*BLOCK_SIZE-1]
    ...all at the same time!
    """

    # STEP 1: "Which chef am I?"
    # tl.program_id(0) gives each chef their unique ID number (0, 1, 2, ...)
    # axis=0 means "the first dimension of work" (we only have one here)
    chef_id = tl.program_id(axis=0)

    # STEP 2: "Which elements am I responsible for?"
    # Chef 0 → elements 0, 1, 2, ..., BLOCK_SIZE-1
    # Chef 1 → elements BLOCK_SIZE, BLOCK_SIZE+1, ..., 2*BLOCK_SIZE-1
    # tl.arange(0, BLOCK_SIZE) = [0, 1, 2, ..., BLOCK_SIZE-1]
    #
    # ANALOGY: Chef 1 walks to position (1 * BLOCK_SIZE) in the freezer
    #          and grabs the next BLOCK_SIZE items
    block_start = chef_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # offsets is now: [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]

    # STEP 3: "Don't go out of bounds!"
    # The last chef might have fewer than BLOCK_SIZE elements to handle
    # (e.g., 1000 elements with BLOCK_SIZE=64: last chef only has 40 elements)
    # mask = True for valid elements, False for "don't touch this"
    mask = offsets < n_elements

    # STEP 4: "Fetch from the freezer" 🧊
    # input_ptr is the ADDRESS where our data starts in GPU memory
    # input_ptr + offsets = addresses of each element we want
    # tl.load() = "go to these addresses and bring the values to my counter"
    # other=0.0 = "if masked out, pretend the value is 0"
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # STEP 5: "Do the cooking" 🍳
    # This is the actual computation — could be anything!
    # Right now: just multiply by 2
    result = x * 2.0

    # STEP 6: "Put it back in the freezer" 🧊
    # Write our result to the output addresses
    tl.store(output_ptr + offsets, result, mask=mask)


# ═════════════════════════════════════════════════════════════════════════════
# LESSON 2: The launcher function
#
# The kernel above runs ON THE GPU.
# We need a normal Python function to CALL it from the outside.
# This is the "head chef" that assigns work to all the GPU chefs!
# ═════════════════════════════════════════════════════════════════════════════

def multiply_by_two(x: torch.Tensor) -> torch.Tensor:
    """
    Python-side launcher. Takes a normal PyTorch tensor, returns a new one
    with every element multiplied by 2.
    """
    # Make sure input is on GPU (our kitchen only works with GPU ingredients!)
    assert x.is_cuda, "Input must be on GPU!"
    assert x.is_contiguous(), "Input must be contiguous in memory!"

    # Allocate output tensor (empty plate to put results on)
    output = torch.empty_like(x)

    # How many elements are we cooking?
    n_elements = x.numel()

    # BLOCK_SIZE = how many elements each "chef" handles at once
    # 1024 is a good default — fills the SRAM counter nicely
    BLOCK_SIZE = 1024

    # How many chefs do we need?
    # If we have 10000 elements and BLOCK_SIZE=1024:
    #   ceil(10000 / 1024) = 10 chefs
    # triton.cdiv = ceiling division (like math.ceil(a/b) but for integers)
    n_chefs = triton.cdiv(n_elements, BLOCK_SIZE)

    # "GRID" = how many chefs are working = how many times the kernel launches
    # It's a tuple because you can have 2D or 3D grids (we'll see that later!)
    grid = (n_chefs,)

    # 🚀 LAUNCH THE KERNEL!
    # Syntax: kernel_function[grid](arguments...)
    # The [...] is the launch config, () is the actual function arguments
    multiply_by_two_kernel[grid](
        x,           # input_ptr  → PyTorch tensors are automatically converted to pointers!
        output,      # output_ptr
        n_elements,  # n_elements
        BLOCK_SIZE=BLOCK_SIZE,   # compile-time constant
    )

    return output


# ═════════════════════════════════════════════════════════════════════════════
# LESSON 3: A slightly more interesting kernel — add two vectors
#
# output[i] = a[i] + b[i]
#
# Same pattern, just two input pointers instead of one!
# ═════════════════════════════════════════════════════════════════════════════

@triton.jit
def add_vectors_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    chef_id = tl.program_id(axis=0)
    offsets = chef_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)  # fetch chunk of a
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)  # fetch chunk of b

    tl.store(output_ptr + offsets, a + b, mask=mask)    # store result


def add_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
    output = torch.empty_like(a)
    n = a.numel()
    grid = (triton.cdiv(n, 1024),)
    add_vectors_kernel[grid](a, b, output, n, BLOCK_SIZE=1024)
    return output


# ═════════════════════════════════════════════════════════════════════════════
# LESSON 4: 2D pointer arithmetic — matrix row normalization
#
# This is the gateway to understanding the rotation kernel later!
# For each ROW of a matrix: output[row] = input[row] / norm(input[row])
#
# Now our "chefs" each handle ONE ROW instead of one scalar block.
# This is a 2D problem:
#   - axis 0 = which row am I? (chef_id)
#   - within my row, I touch all columns
# ═════════════════════════════════════════════════════════════════════════════

@triton.jit
def normalize_rows_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    stride_row,    # how many elements to SKIP to get to the next row
                   # for a contiguous [N, D] matrix: stride_row = D
    BLOCK_COLS: tl.constexpr,  # must be >= n_cols, rounded to power of 2
):
    """
    Each chef handles exactly ONE ROW.
    Chef 0 → row 0
    Chef 1 → row 1
    ...

    Within a row, we load ALL columns at once (they fit in SRAM since D=256).
    """
    row_id = tl.program_id(axis=0)

    # Guard: don't process rows that don't exist
    if row_id >= n_rows:
        return

    # Where does MY row start in memory?
    # ANALOGY: The freezer has rows laid out flat one after another.
    #          Row 0 starts at offset 0
    #          Row 1 starts at offset stride_row (= n_cols for contiguous)
    #          Row 2 starts at offset 2 * stride_row  etc.
    row_start_ptr = input_ptr + row_id * stride_row

    # Column offsets: [0, 1, 2, ..., BLOCK_COLS-1]
    col_offsets = tl.arange(0, BLOCK_COLS)
    mask = col_offsets < n_cols   # mask out padding columns

    # Load the entire row into SRAM (our "counter")
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)

    # Compute the norm (length) of this row
    # norm = sqrt(sum(x^2))
    norm = tl.sqrt(tl.sum(row * row, axis=0))
    norm = tl.maximum(norm, 1e-8)   # avoid division by zero

    # Normalize
    row_normalized = row / norm

    # Write back
    out_row_ptr = output_ptr + row_id * stride_row
    tl.store(out_row_ptr + col_offsets, row_normalized, mask=mask)


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    """Normalize each row of a 2D matrix to unit length."""
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # BLOCK_COLS must be a power of 2 and >= n_cols
    # For head_dim=256: BLOCK_COLS = 256 ✅ (256 is already power of 2!)
    BLOCK_COLS = triton.next_power_of_2(n_cols)

    # One chef per row
    grid = (n_rows,)

    normalize_rows_kernel[grid](
        x, output,
        n_rows, n_cols,
        x.stride(0),    # stride_row: for contiguous tensor = n_cols
        BLOCK_COLS=BLOCK_COLS,
    )
    return output


# ═════════════════════════════════════════════════════════════════════════════
# TESTS — run everything and verify correctness
# ═════════════════════════════════════════════════════════════════════════════

def sep(title=""):
    w = 62
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("\n" + "─" * w)


def check(name, result, expected, tol=1e-3):
    diff = (result.float() - expected.float()).abs().max().item()
    ok = diff < tol
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {status}  {name}  (max diff = {diff:.2e})")
    return ok


def main():
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU found! Triton needs a GPU.")
        return

    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Triton version: {triton.__version__}")

    # ── Test 1: multiply by two ───────────────────────────────────────────────
    sep("Test 1 — multiply_by_two")
    x = torch.randn(10000, device="cuda", dtype=torch.float32)
    result   = multiply_by_two(x)
    expected = x * 2.0
    check("multiply_by_two", result, expected)
    print(f"  Input  [{x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}, ...]")
    print(f"  Output [{result[0]:.4f}, {result[1]:.4f}, {result[2]:.4f}, ...]")

    # ── Test 2: add vectors ───────────────────────────────────────────────────
    sep("Test 2 — add_vectors")
    a = torch.randn(10000, device="cuda")
    b = torch.randn(10000, device="cuda")
    result   = add_vectors(a, b)
    expected = a + b
    check("add_vectors", result, expected)

    # ── Test 3: normalize rows ────────────────────────────────────────────────
    sep("Test 3 — normalize_rows  (gateway to rotation!)")
    # Use exact shape from our KV cache: [N_vectors, head_dim=256]
    N, D = 2048, 256
    x = torch.randn(N, D, device="cuda", dtype=torch.float32)
    result   = normalize_rows(x)
    expected = torch.nn.functional.normalize(x, dim=1)
    check("normalize_rows", result, expected)

    # Verify rows are actually unit length
    norms = result.norm(dim=1)
    print(f"  Row norms after: min={norms.min():.6f}, max={norms.max():.6f}  (should be ~1.0)")

    # ── Speed comparison ──────────────────────────────────────────────────────
    sep("Speed: Triton vs PyTorch normalize_rows")
    import time

    x_big = torch.randn(8192, 256, device="cuda", dtype=torch.float32)

    # Warmup (GPU needs a few runs to hit full speed)
    for _ in range(10):
        _ = normalize_rows(x_big)
        _ = torch.nn.functional.normalize(x_big, dim=1)
    torch.cuda.synchronize()

    # Triton timing
    t0 = time.perf_counter()
    for _ in range(1000):
        normalize_rows(x_big)
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) / 1000 * 1e6  # microseconds

    # PyTorch timing
    t0 = time.perf_counter()
    for _ in range(1000):
        torch.nn.functional.normalize(x_big, dim=1)
    torch.cuda.synchronize()
    t_pytorch = (time.perf_counter() - t0) / 1000 * 1e6

    print(f"  Triton  : {t_triton:.1f} µs")
    print(f"  PyTorch : {t_pytorch:.1f} µs")
    speedup = t_pytorch / t_triton
    print(f"  Speedup : {speedup:.2f}x  {'🔥 Triton wins!' if speedup > 1 else '(PyTorch wins this one, expected for simple ops)'}")

    sep("What we learned!")
    print("""
  1. @triton.jit     → marks a GPU function
  2. tl.program_id() → "which chef am I?"
  3. tl.arange()     → "which elements do I handle?"
  4. tl.load()       → "fetch from GPU memory (the freezer)"
  5. tl.store()      → "write back to GPU memory"
  6. mask            → "don't go out of bounds"
  7. stride          → "how far to jump to get to the next row"
  8. BLOCK_SIZE      → "how big is my chunk" (must be power of 2)

  Next step: fuse normalize + rotate + quantize into ONE kernel!
  That's exactly the TurboQuant compress kernel. 🔥
    """)


if __name__ == "__main__":
    main()