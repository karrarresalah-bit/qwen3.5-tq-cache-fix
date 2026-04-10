# 🚀 qwen3.5-tq-cache-fix (V2 Ultimate)

A custom cache injection to make TurboQuant's 4-bit KV compression fully compatible with Qwen3.5's Hybrid-Attention architecture! 

**V2 Major Update**: We now feature **Position-Aware FLAP Magnitude Pruning** for DeltaNet recurrent states, completely fixing attention passthrough bugs, and offering incredible compression ratios without relying on Triton custom kernels (yet!).

*(Disclaimer: This was largely "vibe coded" into existence! 😅 I built it to solve a specific gap I hit in my own research. The math works, the outputs perfectly match baseline FP16, and the VRAM savings are incredible. If you have optimizations (like a fused Triton kernel), PRs are extremely welcome!)*

## 🚨 The Problem

TurboQuant is incredible at squeezing KV caches down to ~3 or 4 bits using random orthogonal rotation. However, its mathematical foundation assumes standard softmax attention. Qwen3.5 uses a **Hybrid-Attention** architecture (combining standard attention with Gated Delta Networks). Standard TurboQuant fails here because it tries to compress recurrent state matrices that aren't traditional KV pairs.

## ✨ The V2 Solution: The Ultimate Hybrid Cache

This repository provides interceptors to natively hijack the `model.generate()` path:

### `UltimateHybridCache` (Attention + DeltaNet Compression)

This class intercepts the cache and systematically applies customized compression depending on the layer type:

1. **Attention layers (3, 7, 11...)** → TurboQuant 4-bit bit-packing
2. **DeltaNet layers (all other layers)** → Position-Aware FLAP Magnitude Pruning (drops 50% of least important weights while retaining fresh residual memory)

## 📊 Results & Extreme Stress Testing

The hybrid cache's efficiency scales flawlessly. In our extreme long-context stress test (almost 3,000 prompt tokens + 512 generated tokens), the compression ratio holds incredibly strong while preserving perfect textual outputs.

| Target         | Baseline (FP16) Cache | Ultimate V2 Cache | Compression Ratio | Peak VRAM |
| :------------- | :------------------- | :---------------- | :---------------- | :-------- |
| **Short Context** | 53.03 MB       | 8.98 MB         | **5.90x Smaller** | -       |
| **Extreme Stress**| 132.72 MB      | 30.77 MB        | **4.31x Smaller** | 8.73 GB   |

**🔥 ~4.3x - 5.9x KV Cache memory reduction while outputting the exact same text!**

## 🗂️ Repository Structure

We've organized the repository to keep the focus on the production-ready code:

```text
ultimate_qwen_hybrid_cache.py  ← Production module (TurboQuant + FLAP DeltaNet pruning)
benchmark.py                   ← Quick VRAM benchmark (baseline vs ultimate)
stress_test.py                 ← Long-context extreme stress test tool
archive/                       ← Older experiments, calibration scripts, and previous cache versions
ideas.md                       ← Dev notes and future plans
```

## 🛠️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

- `torch >= 2.3.0`
- `transformers >= 4.51.0`
- `bitsandbytes >= 0.44.0`
- `accelerate >= 0.30.0`
- `scipy`

## 🚀 How to Use

Simply import and inject the cache wrapper onto your loaded model before calling `generate`.

```python
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ultimate_qwen_hybrid_cache import inject_ultimate_cache

# Load Qwen3.5 in 4-bit
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

inputs = tokenizer("The ancient city of Aethoria...", return_tensors="pt").to(model.device)

# Inject the Ultimate Cache BEFORE generation
handle, cache_holder = inject_ultimate_cache(model, bits=4, residual_len=64)

with torch.no_grad():
    out = model.generate(
        **inputs, 
        max_new_tokens=150, 
        do_sample=False, 
        return_dict_in_generate=True
    )

# Clean up the hook when done
handle.remove()

print(tokenizer.decode(out.sequences[0], skip_special_tokens=True))

# Precise Memory Tracking
stats = cache_holder["cache"].memory_stats()
print(f"Total VRAM size used: {stats['total_mb']:.2f} MB")
```

## 🔬 Next Steps

With V2 stabilized, the next steps for research focus on:
- Exploring the `Sparsity Death-Slope` to aggressively prune DeltaNet weights up to 90%.
- Fusing operations into a high-performance **Triton Kernel** to bump generation `tok/s` higher without the PyTorch overhead.
