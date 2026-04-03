"""
Step 5 — Delta Cutie Calibration Loop (Magnitude Pruning)
===========================================================
We extract a real DeltaNet recurrent state from Qwen3.5 and 
test its resilience to Magnitude Pruning. 

Goal: Find the maximum sparsity % before Cosine Similarity drops < 0.98
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3.5-9B"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

def sep(title=""):
    w = 65
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("\n" + "─" * w)

def main():
    sep("Step 5 — DeltaNet Sparsity Calibration")
    print("  Loading model to capture a live Delta Cutie...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=BNB_CONFIG, device_map={"": 0}, dtype=torch.float16
    ).eval()

    # 1. Generate a real state
    prompt = "The advanced mathematics of neural networks relies on matrix optimization and"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # Force the model to generate the cache
        outputs = model(**inputs, use_cache=True)
        
    pkv = outputs.past_key_values
    
    # ── The Bulletproof Tensor Sniffer 9000 ──
    def find_tensor(obj):
        """Recursively digs through ANY Python object to find a large Tensor."""
        if isinstance(obj, torch.Tensor) and obj.numel() > 1000:
            return obj
        if isinstance(obj, (list, tuple)):
            for item in obj:
                res = find_tensor(item)
                if res is not None: return res
        if hasattr(obj, "__dict__"):
            for val in vars(obj).values():
                res = find_tensor(val)
                if res is not None: return res
        return None

    delta_state = find_tensor(pkv)

    # Ensure it's a tensor
    if delta_state is None:
        print("  ❌ The Sniffer failed! The cache is completely empty!")
        print(f"  Cache contents: {vars(pkv)}")
        return

    # Ensure it's a tensor
    if not isinstance(delta_state, torch.Tensor):
        print("  ❌ Could not find a valid tensor for the Delta state!")
        print(f"  Cache type was: {type(pkv)}")
        return

    print(f"  ✅ Captured Delta State from Layer 0!")
    print(f"  Shape: {list(delta_state.shape)}")
    print(f"  Range: [{delta_state.min():.4f}, {delta_state.max():.4f}]")
    
    # 2. Calibration Loop
    sep("Magnitude Pruning Stress Test")
    print(f"  {'Target %':<10} {'Threshold (τ)':<15} {'Cosine Sim':<12} {'MSE':<10} {'Verdict'}")
    print("  " + "─" * 58)

    # We flatten it to compute global quantiles and metrics easily
    flat_state = delta_state.flatten().float()
    abs_state = flat_state.abs()

    # Test sparsity levels from 10% to 95%
    test_percentiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    for sparsity in test_percentiles:
        # Calculate the magic threshold number for this percentile
        threshold = torch.quantile(abs_state, sparsity).item()
        
        # Apply pruning: keep only values >= threshold
        mask = abs_state >= threshold
        pruned_state = flat_state * mask
        
        # Measure how badly we damaged the data
        cos_sim = torch.nn.functional.cosine_similarity(
            flat_state.unsqueeze(0), 
            pruned_state.unsqueeze(0)
        ).item()
        
        mse = torch.nn.functional.mse_loss(flat_state, pruned_state).item()
        
        # Determine if it's safe to use
        if cos_sim > 0.99:
            verdict = "✅ Perfect"
        elif cos_sim > 0.95:
            verdict = "🟡 Risky but viable"
        elif cos_sim > 0.85:
            verdict = "⚠️ Degraded"
        else:
            verdict = "❌ Destroyed"

        print(f"  {sparsity*100:>5.0f}%    {threshold:>12.6f}    {cos_sim:>10.4f}   {mse:>8.5f}   {verdict}")

    sep("Next Steps")
    print("  Look for the highest % before the verdict turns yellow or red.")
    print("  That is your magic sparsity target for the DeltaNet cache!\n")

if __name__ == "__main__":
    main()

# ══════════════════════════════════════════════════════════════════════════
# Results from my test run on a 5070 GPU 12GB:
# ══════════════════════════════════════════════════════════════════════════
#  python .\step-05-calibration.py

# ──────────── Step 5 — DeltaNet Sparsity Calibration ─────────────
#   Loading model to capture a live Delta Cutie...
# W0403 14:06:46.186000 14204 site-packages\torch\utils\flop_counter.py:29] triton not found; flop counting will not work for triton kernels
# Fetching 4 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<?, ?it/s]
# Download complete: : 0.00B [00:00, ?B/s]              The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
# Download complete: : 0.00B [00:00, ?B/s]
# Loading weights:   0%|▊                                                                                                                                                                                      | 2/427 [00:00<03:22,  2.10it/s]D:\miniconda3\envs\turboquant_env\Lib\site-packages\bitsandbytes\backends\cuda\ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
#   torch._check_is_size(blocksize)
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 427/427 [00:04<00:00, 94.89it/s]
#   ✅ Captured Delta State from Layer 0!
#   Shape: [1, 8192, 4]
#   Range: [-47.0000, 64.0000]

# ───────────────── Magnitude Pruning Stress Test ─────────────────
#   Target %   Threshold (τ)   Cosine Sim   MSE        Verdict
#   ──────────────────────────────────────────────────────────
#      10%        0.191406        0.9999    0.00124   ✅ Perfect
#      20%        0.394531        0.9995    0.01010   ✅ Perfect
#      30%        0.617188        0.9982    0.03578   ✅ Perfect
#      40%        0.863281        0.9955    0.08993   ✅ Perfect
#      50%        1.148438        0.9903    0.19113   ✅ Perfect
#      60%        1.500000        0.9815    0.36543   🟡 Risky but viable
#      70%        1.921875        0.9665    0.65568   🟡 Risky but viable
#      80%        2.500000        0.9411    1.13702   ⚠️ Degraded
#      90%        3.468750        0.8946    1.98683   ⚠️ Degraded
#      95%        4.718750        0.8481    2.79382   ❌ Destroyed

# ────────────────────────── Next Steps ───────────────────────────
#   Look for the highest % before the verdict turns yellow or red.
#   That is your magic sparsity target for the DeltaNet cache!