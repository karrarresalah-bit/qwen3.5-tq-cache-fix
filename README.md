# 🚀 qwen3.5-tq-cache-fix

A custom cache injection to make TurboQuant's 4-bit KV compression fully compatible with Qwen3.5's Hybrid-Attention architecture — now with **DeltaNet magnitude pruning** for an even deeper compression.

_(Disclaimer: This was largely "vibe coded" into existence! 😅 I built it to solve a specific gap I hit in my own research. The math works and the VRAM savings are real, but constructive feedback and PRs on the code structure are totally welcome!)_

## 🚨 The Problem

TurboQuant is incredible at squeezing KV caches down to ~3 or 4 bits using random orthogonal rotation. However, its mathematical foundation assumes standard softmax attention. Qwen3.5 uses a **Hybrid-Attention** architecture (combining standard attention with Gated Delta Networks). Standard TurboQuant fails here because it tries to compress recurrent state matrices that aren't traditional KV pairs.

## ✨ The Solution

This repository provides two cache classes that act as surgical interceptors:

### `HybridTurboQuantCache` (attention-only compression)

1. **Identifies** standard attention layers (indices 3, 7, 11, 15, 19, 23, 27, 31) and applies true 4-bit bit-packed TurboQuant compression to the KV cache.
2. **Identifies** DeltaNet layers and passes their state matrices through 100% untouched.
3. **Injects** itself cleanly using PyTorch forward pre-hooks — no model weight modifications.

### `UltimateHybridCache` (attention + DeltaNet compression)

Goes further by also compressing the DeltaNet recurrent states:

1. **Attention layers** → 4-bit TurboQuant bit-packing (same as above)
2. **DeltaNet layers** → 50% magnitude pruning via PyTorch sparse tensors (threshold calibrated in Step 5)

## 📊 Results & VRAM Savings

### Attention-Only Compression (Step 4)

**Test Configuration:** Qwen/Qwen3.5-9B (4,691 input tokens, measured directly from cache tensors)

| Configuration               | Tokens/sec | Cache VRAM (Measured) |
| :-------------------------- | :--------- | :-------------------- |
| **Baseline (FP16)**         | 9.5 tok/s  | 176.75 MB             |
| **TurboQuant 4-Bit PACKED** | 7.9 tok/s  | **42.81 MB**          |

**🔥 ~4.1x reduction in KV cache memory!**

### Ultimate Hybrid Cache (Benchmark)

**Test Configuration:** Qwen/Qwen3.5-9B (1,287 input tokens, measured directly from cache tensors)

| Configuration                     | Tokens/sec | Cache VRAM (Measured) |
| :-------------------------------- | :--------- | :-------------------- |
| **Baseline (FP16)**               | 11.7 tok/s | 70.38 MB              |
| **Ultimate (4-bit + 50% Sparse)** | 10.9 tok/s | **13.72 MB**          |

**🔥 ~5.1x reduction — compressing BOTH attention KV and DeltaNet state!**

## 🗂️ Repository Structure

```
ultimate_qwen_hybrid_cache.py  ← Production module (TurboQuant + DeltaNet sparse pruning)
qewn_turboquant_cache.py       ← Attention-only production module (TurboQuant only)
benchmark.py                   ← Head-to-head VRAM benchmark (baseline vs ultimate)
step-01-inspector.py           ← Inspect Qwen3.5 layer types & config
step-02-compressor.py          ← Standalone TurboQuant compress/decompress benchmark
step-03-cache.py               ← First cache attempt (documents the hook-return bug)
step-04-cache-bitpacked.py     ← Fixed cache with true 4-bit bit-packing
step-05-calibration.py         ← DeltaNet magnitude pruning calibration loop
step-06-ultimate-hybrid-cache.py ← Development version of the ultimate cache
step-07-test-ultimate.py       ← Quick smoke test for the ultimate cache
```

Each step script is self-contained, runnable, and has the actual terminal output
appended as comments at the bottom so you can follow the journey.

## 🛠️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

- `torch >= 2.3.0`
- `transformers >= 4.51.0`
- `bitsandbytes >= 0.44.0`
- `accelerate >= 0.30.0`

## 🚀 How to Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ultimate_qwen_hybrid_cache import inject_ultimate_cache

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-9B",
    quantization_config=bnb_config,
    device_map="auto",
)

inputs = tokenizer("Tell me about the universe.", return_tensors="pt").to(model.device)

# Inject the ultimate cache BEFORE generation
handle, cache_holder = inject_ultimate_cache(model, bits=4, residual_len=64)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)

# Clean up the hook when done
handle.remove()

print(tokenizer.decode(out[0], skip_special_tokens=True))

# Inspect cache memory usage
stats = cache_holder["cache"].memory_stats()
print(f"KV cache size: {stats['total_kb'] / 1024:.2f} MB")
```

## 🔬 Step-by-Step Walkthrough

### Step 1 — Architecture Inspector ([step-01-inspector.py](step-01-inspector.py))

Before building anything, we probe the model to find which layers have a KV cache
vs. a recurrent state. Key finding: only 8 of 32 layers are classic attention —
one every 4 layers at indices `[3, 7, 11, 15, 19, 23, 27, 31]`.

```
Total layers    : 32
Attention layers: 8  → indices [3, 7, 11, 15, 19, 23, 27, 31]
DeltaNet layers : 24 → all other indices
config.head_dim = 256
config.num_key_value_heads = 4
```

### Step 2 — Compressor Benchmark ([step-02-compressor.py](step-02-compressor.py))

Tests the TurboQuant math in isolation — no model loading needed.
Verifies that 4-bit compression achieves cosine similarity > 0.99 vs FP16.

```
  bits   cosine        MSE   comp KB   ratio
     4   0.9932   0.000138     280.0    3.66x  ✅ Excellent
     3   0.9706   0.000618     216.0    4.74x  ✅ Great
     2   0.8685   0.003375     152.0    6.74x  ⚠️  Degraded
```

### Step 3 — First Cache Attempt ([step-03-cache.py](step-03-cache.py))

The first attempt at `HybridTurboQuantCache`. The baseline generation works
correctly, but the TurboQuant path crashes with a `RuntimeError` about the
forward pre-hook return value — the hook was returning only `kwargs` instead
of `(args, kwargs)`. This file documents that bug and its traceback.

### Step 4 — Fixed Bit-Packed Cache ([step-04-cache-bitpacked.py](step-04-cache-bitpacked.py))

Fixes the hook return value, adds true 4-bit bit-packing (two 4-bit values per
`uint8` byte), and confirms end-to-end results. Cache sizes are measured
directly from the returned cache tensors (no estimates).

```
Baseline (no TQ):
  short:  30.53 MB Cache size (FP16, measured)
  long : 176.75 MB Cache size (FP16, measured)

TurboQuant 4-bit PACKED:
  short:  2.83 MB Cache size
  long : 42.81 MB Cache size   ← ~4.1× smaller 🔥
```

### Step 5 — DeltaNet Sparsity Calibration ([step-05-calibration.py](step-05-calibration.py))

Captures a live DeltaNet recurrent state from Qwen3.5 and stress-tests
magnitude pruning at increasing sparsity levels. The goal: find the maximum
safe sparsity before cosine similarity drops below 0.99.

```
  Target %   Cosine Sim   Verdict
     10%        0.9999    ✅ Perfect
     30%        0.9982    ✅ Perfect
     50%        0.9903    ✅ Perfect     ← chosen threshold
     60%        0.9815    🟡 Risky but viable
     80%        0.9411    ⚠️ Degraded
     95%        0.8481    ❌ Destroyed
```

### Step 6 — Ultimate Hybrid Cache ([step-06-ultimate-hybrid-cache.py](step-06-ultimate-hybrid-cache.py))

Combines both compression strategies into one cache class:

- **Attention layers**: 4-bit TurboQuant bit-packing
- **DeltaNet layers**: 50% magnitude pruning → PyTorch sparse COO tensors

### Step 7 — Quick Test ([step-07-test-ultimate.py](step-07-test-ultimate.py))

Smoke test that loads Qwen3.5 with the `UltimateHybridCache` injected and
generates a response end-to-end.

### Benchmark ([benchmark.py](benchmark.py))

The definitive head-to-head: baseline FP16 vs. Ultimate Hybrid Cache, with
cache sizes measured by recursively walking every tensor in the cache objects.

```
  Baseline Cache Size:    70.38 MB
  Ultimate Cache Size:    13.72 MB
  Compression Ratio:       5.13x Smaller! 🔥

  Baseline Speed:          11.7 tok/s
  Ultimate Speed:          10.9 tok/s
```
