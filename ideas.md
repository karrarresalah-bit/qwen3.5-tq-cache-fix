# 🚀 Refined Calibration & Sparsity Implementation

## 📊 Calibration Dataset Mix (256-512 samples)
- [ ] **Collect 60-70% General Pre-training Data** (~240 samples)
    - [x] Focus: Pile or RedPajama (fallback: Clean C4)
    - [ ] Goal: Ensure DeltaNet stability and state propagation across diverse domains (web, books, code, science).
- [ ] **Collect 20% Reasoning/Arithmetic Traces** (~80 samples)
    - [x] Focus: GSM8K (train split)
    - [ ] Goal: Protect logical chains and ensure recurrent layers carry precise information.
- [ ] **Collect 10-20% Long-coherent Text** (~80 samples)
    - [x] Focus: WikiText-2 or BookCorpus slices
    - [ ] Goal: Stress test gated decay and delta-rule updates over distance.

## 🛠️ `step-05-calibration.py` Upgrades
- [ ] **Implement `get_calibration_dataset` function**
    - [x] Link `num_samples=400` and `seq_len=1024` (or 2048 for long-state tests).
    - [x] Implement streaming load and shuffling for Pile, GSM8K, and WikiText.
    - [x] Concatenate and shuffle the final mixed dataset.
    - [x] Integrate Qwen tokenizer logic (truncation/padding).
- [ ] **Enhance Calibration Loop Logic**
    - [x] Set up forward hooks to capture DeltaNet states (ssm_*, gates, recurrent state).
    - [x] **Implement FLAP-style importance:** Compute `weight_norm × activation_variance`.
    - [x] Accumulate stats across the mix with `batch_size=1`.
    - [x] Prune at 50% threshold and measure cosine similarity on reconstructed vs. original states.

## 🧪 UltimateHybridCache Optimization
- [ ] **Implement Position-Aware Pruning**
    - [ ] Apply **40% sparsity** to early DeltaNet layers (0-11) to protect foundational state.
    - [ ] Apply **50-60% sparsity** to later layers where state is more robust.
    - [ ] Validate cosine similarity metrics across positions.

---

### 💾 Reference: Calibration Implementation Pseudocode

```python
from datasets import load_dataset, concatenate_datasets
import torch

# Build mixed calibration set
def get_calibration_dataset(num_samples=400, seq_len=1024):
    pile = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True).shuffle(seed=42).take(240)
    gsm = load_dataset("gsm8k", "main", split="train").take(80)
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").take(80)
    
    dataset = concatenate_datasets([pile, gsm, wikitext]).shuffle(seed=42)
    # Tokenize with Qwen tokenizer, truncate/pad to seq_len, return input_ids
    return tokenized_dataset

# In your calibration loop:
calib_data = get_calibration_dataset(num_samples=400, seq_len=1024)
for batch in calib_data:
    with torch.no_grad():
        outputs = model(**batch)  # Use forward hook to capture DeltaNet states
        # Collect activations for Delta-specific tensors (ssm_*, gates, recurrent state)
        # Compute importance: weight_norm × activation_variance (FLAP-style)
        # Prune at 50% threshold, measure cosine sim on reconstructed state vs original
```