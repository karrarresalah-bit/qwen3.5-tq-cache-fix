"""
Step 1 — Qwen3.5-9B Architecture Inspector
============================================
Before we build HybridTurboQuantCache, let's SEE the model from the inside:
  - What layer types exist?
  - Which layers have a KV cache (attention) vs a state matrix (DeltaNet)?
  - What are the head dims, num heads, etc.?

This tells us exactly WHAT to compress and WHAT to leave alone.

No generation, no TurboQuant — just inspection. Fast!
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3.5-9B"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

def sep(title=""):
    w = 62
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("\n" + "─" * w)


def inspect_model(model):
    sep("Top-level model structure")
    for name, module in model.named_children():
        print(f"  {name}: {type(module).__name__}")

    # Drill into the transformer layers
    # Qwen3.5 structure: model.model.layers[i]
    try:
        layers = model.model.layers
    except AttributeError:
        print("Could not find model.model.layers — check structure above")
        return

    sep(f"Layer-by-layer breakdown ({len(layers)} total layers)")

    attention_indices  = []
    deltanet_indices   = []
    other_indices      = []

    for i, layer in enumerate(layers):
        # Each decoder layer has a self_attn sub-module
        # Let's see what type it is
        attn = getattr(layer, "self_attn", None)
        attn_type = type(attn).__name__ if attn is not None else "None"

        # Check for DeltaNet-specific attributes
        is_deltanet  = hasattr(attn, "q_proj") and hasattr(attn, "key_dim")  # DeltaNet uses key_dim
        is_attention = hasattr(attn, "q_proj") and hasattr(attn, "num_heads") and not is_deltanet

        # Also check by class name
        cname = attn_type.lower()
        if "delta" in cname or "linear" in cname:
            tag = "🔷 DeltaNet (linear attn, STATE matrix, skip TQ)"
            deltanet_indices.append(i)
        elif "attention" in cname or "attn" in cname:
            tag = "🟡 Attention  (classic KV cache, COMPRESS with TQ)"
            attention_indices.append(i)
        else:
            tag = f"❓ Unknown: {attn_type}"
            other_indices.append(i)

        print(f"  Layer {i:2d}: {tag}")

        # For attention layers, print extra details
        if "attention" in cname or "attn" in cname:
            num_heads    = getattr(attn, "num_heads",    "?")
            num_kv_heads = getattr(attn, "num_key_value_heads", "?")
            head_dim     = getattr(attn, "head_dim",     "?")
            print(f"           num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

    sep("Summary")
    print(f"  Total layers    : {len(layers)}")
    print(f"  Attention layers: {len(attention_indices)}  → indices {attention_indices}")
    print(f"  DeltaNet layers : {len(deltanet_indices)}  → indices {deltanet_indices}")
    if other_indices:
        print(f"  Unknown layers  : {len(other_indices)}  → indices {other_indices}")

    sep("Pattern detection")
    # Try to detect the repeating block pattern
    # e.g. [D, D, D, A] repeated
    if attention_indices and deltanet_indices:
        # Find spacing between attention layers
        gaps = [attention_indices[0]] + [
            attention_indices[i+1] - attention_indices[i]
            for i in range(len(attention_indices) - 1)
        ]
        print(f"  Gaps between attention layers: {gaps}")
        if len(set(gaps)) == 1:
            block = gaps[0]
            print(f"  ✓ Regular pattern detected: 1 attention every {block} layers")
            print(f"    = [{block-1} DeltaNet → 1 Attention] × {len(attention_indices)}")
        else:
            print(f"  ✗ Irregular pattern — will need per-layer detection")

    sep("Cache-relevant config")
    cfg = model.config
    interesting = [
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_size",
        "attn_layer_period",   # Qwen3.5-specific: how often attention appears
        "attn_layer_offset",   # Qwen3.5-specific: where in the block attention sits
    ]
    for attr in interesting:
        val = getattr(cfg, attr, "not found")
        print(f"  config.{attr} = {val}")

    sep("HybridCache clues")
    # Let's also look at what the model's _cache_class or default cache is
    cache_cls = getattr(model, "_cache_class", None)
    print(f"  model._cache_class = {cache_cls}")
    get_cache = getattr(model, "_get_cache", None)
    print(f"  model._get_cache   = {get_cache}")


def main():
    sep("Loading model (4-bit, inspection only)")
    print(f"  Model: {MODEL_ID}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map={"": 0},
        dtype=torch.float16,
    )
    model.eval()
    print("  ✓ Loaded!")

    inspect_model(model)

    sep("Done")
    print("\n  Now we know exactly which layers to compress")
    print("  and which to leave alone. Time to build HybridTurboQuantCache! 🔥\n")


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# Results for Qwen3.5-9B (4-bit weights):
# -----------------------------------------------------------------------------
# ❯ python .\inspector.py 

# ─────────── Loading model (4-bit, inspection only) ───────────
#   Model: Qwen/Qwen3.5-9B
# W0402 17:31:17.565000 29884 site-packages\torch\utils\flop_counter.py:29] triton not found; flop counting will not work for triton kernels
# Fetching 4 files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<?, ?it/s]
# Download complete: : 0.00B [00:00, ?B/s]                                                                                                                                          | 0/4 [00:00<?, ?it/s] 
# The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
# Loading weights:   0%|▋                                                                                                                                                 | 2/427 [00:00<03:24,  2.08it/s]D:\miniconda3\envs\turboquant_env\Lib\site-packages\bitsandbytes\backends\cuda\ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
#   torch._check_is_size(blocksize)
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 427/427 [00:04<00:00, 94.10it/s]
#   ✓ Loaded!

# ───────────────── Top-level model structure ──────────────────
#   model: Qwen3_5TextModel
#   lm_head: Linear

# ───────── Layer-by-layer breakdown (32 total layers) ─────────
#   Layer  0: ❓ Unknown: None
#   Layer  1: ❓ Unknown: None
#   Layer  2: ❓ Unknown: None
#   Layer  3: 🟡 Attention  (classic KV cache, COMPRESS with TQ)
#            num_heads=?, num_kv_heads=?, head_dim=256
#   Layer  4: ❓ Unknown: None
#   Layer  5: ❓ Unknown: None
#   Layer  6: ❓ Unknown: None
#   Layer  7: 🟡 Attention  (classic KV cache, COMPRESS with TQ)
#            num_heads=?, num_kv_heads=?, head_dim=256
#   Layer  8: ❓ Unknown: None
#   Layer  9: ❓ Unknown: None
#   Layer 10: ❓ Unknown: None
#   Layer 11: 🟡 Attention  (classic KV cache, COMPRESS with TQ)
#            num_heads=?, num_kv_heads=?, head_dim=256
#   Layer 12: ❓ Unknown: None
#   Layer 13: ❓ Unknown: None
#   Layer 14: ❓ Unknown: None
#   Layer 15: 🟡 Attention  (classic KV cache, COMPRESS with TQ)
#            num_heads=?, num_kv_heads=?, head_dim=256
#   Layer 16: ❓ Unknown: None
#   Layer 17: ❓ Unknown: None
#   Layer 18: ❓ Unknown: None
#   Layer 19: 🟡 Attention  (classic KV cache, COMPRESS with TQ)
#            num_heads=?, num_kv_heads=?, head_dim=256
#   Layer 20: ❓ Unknown: None
#   Layer 21: ❓ Unknown: None
#   Layer 22: ❓ Unknown: None
#   Layer 23: 🟡 Attention  (classic KV cache, COMPRESS with TQ)
#            num_heads=?, num_kv_heads=?, head_dim=256
#   Layer 24: ❓ Unknown: None
#   Layer 25: ❓ Unknown: None
#   Layer 26: ❓ Unknown: None
#   Layer 27: 🟡 Attention  (classic KV cache, COMPRESS with TQ)
#            num_heads=?, num_kv_heads=?, head_dim=256
#   Layer 28: ❓ Unknown: None
#   Layer 29: ❓ Unknown: None
#   Layer 30: ❓ Unknown: None
#   Layer 31: 🟡 Attention  (classic KV cache, COMPRESS with TQ)
#            num_heads=?, num_kv_heads=?, head_dim=256

# ────────────────────────── Summary ───────────────────────────
#   Total layers    : 32
#   Attention layers: 8  → indices [3, 7, 11, 15, 19, 23, 27, 31]
#   DeltaNet layers : 0  → indices []
#   Unknown layers  : 24  → indices [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]

# ───────────────────── Pattern detection ──────────────────────

# ─────────────────── Cache-relevant config ────────────────────
#   config.num_hidden_layers = 32
#   config.num_attention_heads = 16
#   config.num_key_value_heads = 4
#   config.head_dim = 256
#   config.hidden_size = 4096
#   config.attn_layer_period = not found
#   config.attn_layer_offset = not found

# ───────────────────── HybridCache clues ──────────────────────
#   model._cache_class = None
#   model._get_cache   = None

# ──────────────────────────── Done ────────────────────────────

#   Now we know exactly which layers to compress
#   and which to leave alone. Time to build HybridTurboQuantCache! 🔥