"""
Step 3 — HybridTurboQuantCache for Qwen3.5-9B
===============================================
We build a cache class that:
  ✅ Makes Qwen3.5 happy (has all methods it expects)
  ✅ Compresses attention KV pairs with TurboQuant
  ✅ Leaves DeltaNet state matrices 100% untouched

Strategy: since HybridCache was removed in transformers v5,
we DON'T subclass it. Instead we build our own cache from scratch
that mimics the exact interface Qwen3.5's modeling code expects,
discovered by reading the error traceback from step 0.

The two methods Qwen3.5 MUST find on our cache:
  - .has_previous_state(layer_idx) → bool
  - .update(key, value, layer_idx, cache_kwargs) → (key, value)

We intercept .update() for attention layers and compress the KV there.
DeltaNet layers call .has_previous_state() — we return their state untouched.

Run:
    python step3_cache.py
"""

import torch
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3.5-9B"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# From step 1: these are the ONLY layers with a classic KV cache
ATTENTION_LAYER_INDICES = {3, 7, 11, 15, 19, 23, 27, 31}
HEAD_DIM = 256


# ═════════════════════════════════════════════════════════════════════════════
# TurboQuant compressor (from step 2, cleaned up)
# ═════════════════════════════════════════════════════════════════════════════

def make_rotation_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(dim, dim))
    return Q

def make_codebook(bits: int) -> torch.Tensor:
    return torch.linspace(-3.0, 3.0, 2 ** bits)

def compress_tensor(x: torch.Tensor, rotation: torch.Tensor, codebook: torch.Tensor):
    """
    Compress a KV tensor of arbitrary shape [..., head_dim].
    Works on any prefix shape — heads, batch, sequence, all handled.
    Returns (indices, scales, mus, stds) — everything needed to decompress.
    """
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    # Flatten everything except the last dim: [..., head_dim] → [N, head_dim]
    flat = x.reshape(-1, head_dim).float()
    N = flat.shape[0]

    rotation = rotation.to(flat.device)
    codebook = codebook.to(flat.device)

    # Normalize each vector to unit length, rotate, normalize to codebook range
    norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)   # [N, 1]
    unit  = flat / norms                                        # [N, head_dim]
    rotated = unit @ rotation.T                                 # [N, head_dim]

    mu  = rotated.mean(dim=1, keepdim=True)                    # [N, 1]
    std = rotated.std(dim=1, keepdim=True).clamp(min=1e-8)     # [N, 1]
    normed = (rotated - mu) / std                              # [N, head_dim]

    # Quantize: find nearest codebook entry for each value
    # normed: [N, head_dim, 1]  codebook: [1, 1, n_levels]
    dist    = (normed.unsqueeze(2) - codebook.reshape(1, 1, -1)).abs()
    indices = dist.argmin(dim=2).to(torch.uint8)               # [N, head_dim]

    # Reshape everything back to original prefix shape
    prefix = orig_shape[:-1]
    return (
        indices.reshape(*prefix, head_dim),
        norms.reshape(*prefix, 1),
        mu.reshape(*prefix, 1),
        std.reshape(*prefix, 1),
    )

def decompress_tensor(indices, norms, mu, std,
                      rotation: torch.Tensor,
                      codebook: torch.Tensor,
                      target_dtype=torch.float16) -> torch.Tensor:
    """Decompress back to a KV tensor."""
    orig_shape = indices.shape
    head_dim = orig_shape[-1]
    flat_idx   = indices.reshape(-1, head_dim).long()
    flat_norms = norms.reshape(-1, 1).float()
    flat_mu    = mu.reshape(-1, 1).float()
    flat_std   = std.reshape(-1, 1).float()

    rotation = rotation.to(flat_idx.device)
    codebook = codebook.to(flat_idx.device)

    # Look up → undo range normalization → rotate back → restore norm
    looked_up = codebook[flat_idx]                        # [N, head_dim]
    unscaled  = looked_up * flat_std + flat_mu            # undo normalization
    unrotated = unscaled @ rotation                        # rotate backwards
    restored  = unrotated * flat_norms                    # restore original length

    return restored.reshape(orig_shape).to(target_dtype)


# ═════════════════════════════════════════════════════════════════════════════
# The Cache Class
# ═════════════════════════════════════════════════════════════════════════════

class HybridTurboQuantCache:
    """
    A cache for Qwen3.5's hybrid architecture that:
      - Stores DeltaNet states exactly as a normal HybridCache would
      - Compresses attention KV pairs with TurboQuant on the fly

    Qwen3.5 calls two things on this cache:
      1. cache.update(k, v, layer_idx, cache_kwargs)   ← every layer
      2. cache.has_previous_state(layer_idx)            ← DeltaNet layers only

    We intercept update() for attention layers and compress there.
    Everything else is passed through to the underlying real cache.
    """

    def __init__(self, real_cache, bits: int = 4, residual_len: int = 64):
        """
        Args:
            real_cache  : the actual HybridCache-like object Qwen3.5 built
            bits        : quantization bits (4 recommended)
            residual_len: keep this many most-recent tokens uncompressed (fp16)
                          protects quality during active generation
        """
        self._real   = real_cache       # delegate everything DeltaNet-related here
        self.bits    = bits
        self.residual_len = residual_len

        # Build TurboQuant tools — one rotation matrix, one codebook
        self.rotation = make_rotation_matrix(HEAD_DIM)
        self.codebook = make_codebook(bits)

        # Storage for compressed attention KV
        # Dict: layer_idx → {"k_idx", "k_norms", "k_mu", "k_std",
        #                     "v_idx", "v_norms", "v_mu", "v_std",
        #                     "k_residual", "v_residual"}
        self._compressed = {}

        # Stats
        self.n_compressed = 0
        self.n_passthrough = 0

    # ── The two methods Qwen3.5 MUST find ────────────────────────────────────

    def has_previous_state(self, layer_idx: int = None) -> bool:
        """Called by DeltaNet layers. Delegate to real cache."""
        return self._real.has_previous_state(layer_idx)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        Called by EVERY layer on every forward pass.
        - DeltaNet layers (not in ATTENTION_LAYER_INDICES): pass straight through
        - Attention layers: compress old tokens, keep recent ones hot in fp16
        """
        if layer_idx not in ATTENTION_LAYER_INDICES:
            # DeltaNet layer — not our business, delegate completely
            self.n_passthrough += 1
            return self._real.update(key_states, value_states, layer_idx, cache_kwargs)

        # ── Attention layer — we handle this! ─────────────────────────────────
        self.n_compressed += 1

        if layer_idx not in self._compressed:
            # First time seeing this layer — just store as-is (residual window)
            self._compressed[layer_idx] = {
                "k_residual": key_states,
                "v_residual": value_states,
                "k_idx": None, "k_norms": None, "k_mu": None, "k_std": None,
                "v_idx": None, "v_norms": None, "v_mu": None, "v_std": None,
            }
            return key_states, value_states

        store = self._compressed[layer_idx]

        # Append new tokens to residual window
        store["k_residual"] = torch.cat([store["k_residual"], key_states],   dim=2)
        store["v_residual"] = torch.cat([store["v_residual"], value_states], dim=2)

        seq_len = store["k_residual"].shape[2]

        # If residual window is full → compress the older half, keep recent tokens hot
        if seq_len > self.residual_len:
            n_to_compress = seq_len - self.residual_len

            k_old = store["k_residual"][:, :, :n_to_compress, :]  # tokens to compress
            v_old = store["v_residual"][:, :, :n_to_compress, :]

            # Compress!
            k_idx, k_norms, k_mu, k_std = compress_tensor(k_old, self.rotation, self.codebook)
            v_idx, v_norms, v_mu, v_std = compress_tensor(v_old, self.rotation, self.codebook)

            # Accumulate compressed chunks
            store["k_idx"]   = k_idx   if store["k_idx"]   is None else torch.cat([store["k_idx"],   k_idx],   dim=2)
            store["k_norms"] = k_norms if store["k_norms"] is None else torch.cat([store["k_norms"], k_norms], dim=2)
            store["k_mu"]    = k_mu    if store["k_mu"]    is None else torch.cat([store["k_mu"],    k_mu],    dim=2)
            store["k_std"]   = k_std   if store["k_std"]   is None else torch.cat([store["k_std"],   k_std],   dim=2)

            store["v_idx"]   = v_idx   if store["v_idx"]   is None else torch.cat([store["v_idx"],   v_idx],   dim=2)
            store["v_norms"] = v_norms if store["v_norms"] is None else torch.cat([store["v_norms"], v_norms], dim=2)
            store["v_mu"]    = v_mu    if store["v_mu"]    is None else torch.cat([store["v_mu"],    v_mu],    dim=2)
            store["v_std"]   = v_std   if store["v_std"]   is None else torch.cat([store["v_std"],   v_std],   dim=2)

            # Trim residual to only recent tokens
            store["k_residual"] = store["k_residual"][:, :, n_to_compress:, :]
            store["v_residual"] = store["v_residual"][:, :, n_to_compress:, :]

        # Build full KV for attention computation:
        # decompress old tokens + concatenate with hot residual
        if store["k_idx"] is not None:
            k_old_decomp = decompress_tensor(
                store["k_idx"], store["k_norms"], store["k_mu"], store["k_std"],
                self.rotation, self.codebook, target_dtype=store["k_residual"].dtype
            )
            v_old_decomp = decompress_tensor(
                store["v_idx"], store["v_norms"], store["v_mu"], store["v_std"],
                self.rotation, self.codebook, target_dtype=store["v_residual"].dtype
            )
            full_k = torch.cat([k_old_decomp, store["k_residual"]], dim=2)
            full_v = torch.cat([v_old_decomp, store["v_residual"]], dim=2)
        else:
            full_k = store["k_residual"]
            full_v = store["v_residual"]

        return full_k, full_v

    # ── Delegate everything else to the real cache ────────────────────────────
    # Qwen3.5's generate() also accesses these attributes/methods

    def __getattr__(self, name):
        # Only called if attribute not found on self — delegate to real cache
        return getattr(self._real, name)

    def memory_stats(self):
        """How much VRAM are we saving?"""
        compressed_bytes = 0
        residual_bytes   = 0
        for store in self._compressed.values():
            if store["k_idx"] is not None:
                compressed_bytes += store["k_idx"].numel()    # uint8 = 1 byte
                compressed_bytes += store["k_norms"].numel() * 4  # float32
                compressed_bytes += store["k_mu"].numel()    * 4
                compressed_bytes += store["k_std"].numel()   * 4
                compressed_bytes += store["v_idx"].numel()
                compressed_bytes += store["v_norms"].numel() * 4
                compressed_bytes += store["v_mu"].numel()    * 4
                compressed_bytes += store["v_std"].numel()   * 4
            if store["k_residual"] is not None:
                residual_bytes += store["k_residual"].numel() * 2  # fp16
                residual_bytes += store["v_residual"].numel() * 2
        return {
            "compressed_kb": compressed_bytes / 1024,
            "residual_kb":   residual_bytes   / 1024,
            "total_kb":      (compressed_bytes + residual_bytes) / 1024,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Injection helper
# ═════════════════════════════════════════════════════════════════════════════

def inject_turbo_cache(model, bits: int = 4, residual_len: int = 64):
    original_generate = model.generate.__func__

    def patched_generate(self_model, *args, **kwargs):
        result = original_generate(self_model, *args, **kwargs)
        return result

    turbo_cache_holder = {"cache": None, "injected": False}

    def forward_hook(module, input, kwargs_hook):
        if not turbo_cache_holder["injected"] and "past_key_values" in kwargs_hook:
            real_cache = kwargs_hook["past_key_values"]
            if real_cache is not None and not isinstance(real_cache, HybridTurboQuantCache):
                wrapped = HybridTurboQuantCache(real_cache, bits=bits, residual_len=residual_len)
                kwargs_hook["past_key_values"] = wrapped
                turbo_cache_holder["cache"]    = wrapped
                turbo_cache_holder["injected"] = True
                
        # THE FIX: Return both the positional inputs and the modified kwargs
        return input, kwargs_hook

    handle = model.model.register_forward_pre_hook(forward_hook, with_kwargs=True)
    return handle, turbo_cache_holder


# ═════════════════════════════════════════════════════════════════════════════
# Test — run baseline vs TurboQuant and compare
# ═════════════════════════════════════════════════════════════════════════════

def sep(title=""):
    w = 62
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("\n" + "─" * w)


# A long prompt to stress the cache
_LORE = (
    "The ancient city of Aethoria stood at the edge of the Silver Sea. "
    "Its towers of enchanted stone hummed softly in the ocean wind. "
    "For centuries the scholars of Aethoria devoted their lives to unraveling the mysteries of time. "
    "Beneath the city lay the Vault of Echoes, where memories of the dead were crystallized and stored. "
    "Every citizen who passed left behind a shard containing their final thoughts and feelings. "
    "The keepers of the Vault were a secretive order known as the Remembrancers. "
    "They alone could read the shards and speak with the voices of the departed. "
) * 40

TEST_PROMPTS = [
    "Write a short poem about a samurai watching the sunrise.",
    _LORE + "\n\nSummarize the key themes in this passage.",
]


def run_test(model, tokenizer, prompt, label, bits=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_input = inputs["input_ids"].shape[-1]

    handle, cache_holder = None, None
    if bits is not None:
        handle, cache_holder = inject_turbo_cache(model, bits=bits, residual_len=64)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if handle is not None:
        handle.remove()

    generated = out[0][n_input:]
    text      = tokenizer.decode(generated, skip_special_tokens=True)
    tps       = len(generated) / elapsed
    peak_mb   = torch.cuda.max_memory_allocated() / 1e6

    result = {
        "label":   label,
        "text":    text,
        "tps":     tps,
        "peak_mb": peak_mb,
        "n_input": n_input,
        "tokens":  len(generated),
    }

    if cache_holder and cache_holder["cache"]:
        result["cache_stats"] = cache_holder["cache"].memory_stats()
        result["n_compressed"] = cache_holder["cache"].n_compressed
        result["n_passthrough"] = cache_holder["cache"].n_passthrough

    return result


def main():
    sep("Step 3 — HybridTurboQuantCache Test")
    print(f"  Model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    configs = [
        ("Baseline (no TQ)", None),
        ("TurboQuant 4-bit", 4),
        ("TurboQuant 3-bit", 3),
    ]

    all_results = {cfg: [] for cfg, _ in configs}

    for cfg_label, bits in configs:
        sep(cfg_label)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=BNB_CONFIG,
            device_map={"": 0},
            dtype=torch.float16,
        )
        model.eval()

        for i, prompt in enumerate(TEST_PROMPTS):
            p_label = ["short", "long-context"][i]
            n_tok   = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]
            print(f"\n  Prompt {i+1} ({p_label}, {n_tok} input tokens)")

            r = run_test(model, tokenizer, prompt, cfg_label, bits=bits)
            all_results[cfg_label].append(r)

            print(f"  → {r['tps']:.1f} tok/s | peak {r['peak_mb']:.0f} MB | {r['tokens']} tokens")
            if "cache_stats" in r:
                s = r["cache_stats"]
                print(f"     compressed: {s['compressed_kb']:.0f} KB | "
                      f"residual: {s['residual_kb']:.0f} KB | "
                      f"total: {s['total_kb']:.0f} KB")
                print(f"     attention calls compressed: {r['n_compressed']} | "
                      f"DeltaNet passthrough: {r['n_passthrough']}")
            print(f"  Output: {r['text'][:120].strip()}…")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    sep("SUMMARY")
    print(f"\n{'Config':<22} {'prompt':>8} {'tok/s':>7} {'peak MB':>9} {'input tok':>10}")
    print("─" * 60)
    for cfg_label, results in all_results.items():
        for i, r in enumerate(results):
            p = ["short", "long "][i]
            print(f"{cfg_label:<22} {p:>8} {r['tps']:>7.1f} {r['peak_mb']:>9.0f} {r['n_input']:>10}")

    sep("QUALITY CHECK — short prompt output")
    for cfg_label, results in all_results.items():
        print(f"\n[{cfg_label}]")
        print(results[0]["text"][:250].strip())

    sep()
    print("\n  🔥 If it runs without crashing → we solved the HybridCache gap!")
    print("  ✅ If peak MB is lower for TQ on long prompt → compression works!")
    print("  ✅ If outputs match → quality preserved!\n")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------------
# Results came with error
# -----------------------------------------------------------------------------------
# ❯ python .\step-03-cache.py     

# ──────────── Step 3 — HybridTurboQuantCache Test ─────────────
#   Model: Qwen/Qwen3.5-9B

# ────────────────────── Baseline (no TQ) ──────────────────────
# W0402 17:35:02.898000 29984 site-packages\torch\utils\flop_counter.py:29] triton not found; flop counting will not work for triton kernels
# Fetching 4 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 2660.94it/s]
# Download complete: : 0.00B [00:00, ?B/s]              The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
# Download complete: : 0.00B [00:00, ?B/s]
# Loading weights:   0%|▋                                                                                                                                                 | 2/427 [00:00<03:25,  2.07it/s]D:\miniconda3\envs\turboquant_env\Lib\site-packages\bitsandbytes\backends\cuda\ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
#   torch._check_is_size(blocksize)
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 427/427 [00:04<00:00, 92.92it/s]

#   Prompt 1 (short, 12 input tokens)
# Setting `pad_token_id` to `eos_token_id`:248044 for open-end generation.
# D:\miniconda3\envs\turboquant_env\Lib\site-packages\bitsandbytes\backends\cuda\ops.py:468: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
#   torch._check_is_size(blocksize)
#   → 12.0 tok/s | peak 7886 MB | 150 tokens
#   Output: <think>
# Thinking Process:

# 1.  **Analyze the Request:**
#     *   **Topic:** A samurai watching the sunrise.
#     *   **F…

#   Prompt 2 (long-context, 4691 input tokens)
# Setting `pad_token_id` to `eos_token_id`:248044 for open-end generation.
#   → 9.1 tok/s | peak 11638 MB | 150 tokens
#   Output: <think>
# Thinking Process:

# 1.  **Analyze the Request:**
#     *   Input: A text passage.
#     *   Task: Summarize the key…

# ────────────────────── TurboQuant 4-bit ──────────────────────
# Fetching 4 files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<?, ?it/s]
# Download complete: : 0.00B [00:00, ?B/s]                                                                                                                                          | 0/4 [00:00<?, ?it/s] 
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 427/427 [00:04<00:00, 92.66it/s]

#   Prompt 1 (short, 12 input tokens)
# Setting `pad_token_id` to `eos_token_id`:248044 for open-end generation.
# Traceback (most recent call last):
#   File "D:\turboquant-torch\step-03-cache.py", line 465, in <module>
#     main()
#   File "D:\turboquant-torch\step-03-cache.py", line 427, in main
#     r = run_test(model, tokenizer, prompt, cfg_label, bits=bits)
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "D:\turboquant-torch\step-03-cache.py", line 361, in run_test
#     out = model.generate(
#           ^^^^^^^^^^^^^^^
#   File "D:\miniconda3\envs\turboquant_env\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
#     return func(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "D:\miniconda3\envs\turboquant_env\Lib\site-packages\transformers\generation\utils.py", line 2543, in generate
#     result = decoding_method(
#              ^^^^^^^^^^^^^^^^
#   File "D:\miniconda3\envs\turboquant_env\Lib\site-packages\transformers\generation\utils.py", line 2736, in _sample
#     outputs = self._prefill(
#               ^^^^^^^^^^^^^^
#   File "D:\miniconda3\envs\turboquant_env\Lib\site-packages\transformers\generation\utils.py", line 3768, in _prefill
#     return self(**model_inputs, return_dict=True)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "D:\miniconda3\envs\turboquant_env\Lib\site-packages\torch\nn\modules\module.py", line 1779, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "D:\miniconda3\envs\turboquant_env\Lib\site-packages\torch\nn\modules\module.py", line 1790, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "D:\miniconda3\envs\turboquant_env\Lib\site-packages\transformers\utils\generic.py", line 876, in wrapper
#     output = func(self, *args, **kwargs)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "D:\miniconda3\envs\turboquant_env\Lib\site-packages\transformers\models\qwen3_5\modeling_qwen3_5.py", line 1740, in forward
#     outputs: BaseModelOutputWithPast = self.model(
#                                        ^^^^^^^^^^^
#   File "D:\miniconda3\envs\turboquant_env\Lib\site-packages\torch\nn\modules\module.py", line 1779, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "D:\miniconda3\envs\turboquant_env\Lib\site-packages\torch\nn\modules\module.py", line 1885, in _call_impl
#     return inner()
#            ^^^^^^^
#   File "D:\miniconda3\envs\turboquant_env\Lib\site-packages\torch\nn\modules\module.py", line 1817, in inner
#     raise RuntimeError(
# RuntimeError: forward pre-hook must return None or a tuple of (new_args, new_kwargs), but got {'input_ids': tensor([[ 7734,   264,  2716, 31708,   883,   264,  9667, 46314,  9799,   279,
#          61665,    13]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0'), 'position_ids': tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]], device='cuda:0'), 'past_key_values': <__main__.HybridTurboQuantCache object at 0x000002376DBA1610>, 'inputs_embeds': None, 'use_cache': True}.