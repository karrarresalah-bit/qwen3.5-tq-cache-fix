"""
Step 4 — True 4-bit Bit-Packing Hybrid Cache
===============================================
Now we compress the indices down to TRUE 4-bit sizes by packing 
two 4-bit integers into a single 8-bit byte.

Run:
    python step-04-cache-bitpacked.py
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

ATTENTION_LAYER_INDICES = {3, 7, 11, 15, 19, 23, 27, 31}
HEAD_DIM = 256


# ═════════════════════════════════════════════════════════════════════════════
# NEW: Bit-Packing Helpers
# ═════════════════════════════════════════════════════════════════════════════

def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Packs a uint8 tensor of values (0-15) into half the space."""
    # indices shape: [..., D] -> packed shape: [..., D/2]
    even = indices[..., 0::2].to(torch.uint8)
    odd  = indices[..., 1::2].to(torch.uint8)
    
    # Shift even bits left by 4, and bitwise OR with odd bits
    # Example: even=5 (00000101), odd=3 (00000011)
    # even << 4 = 01010000
    # (even << 4) | odd = 01010011
    packed = (even << 4) | odd
    return packed

def unpack_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpacks a packed 4-bit tensor back to original size."""
    even = packed >> 4
    odd  = packed & 0x0F
    
    # Interleave them back together
    unpacked = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8, device=packed.device)
    unpacked[..., 0::2] = even
    unpacked[..., 1::2] = odd
    return unpacked

# ═════════════════════════════════════════════════════════════════════════════
# TurboQuant tools
# ═════════════════════════════════════════════════════════════════════════════

def make_rotation_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(dim, dim))
    return Q

def make_codebook(bits: int) -> torch.Tensor:
    return torch.linspace(-3.0, 3.0, 2 ** bits)

def compress_tensor(x: torch.Tensor, rotation: torch.Tensor, codebook: torch.Tensor):
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    flat = x.reshape(-1, head_dim).float()
    N = flat.shape[0]

    rotation = rotation.to(flat.device)
    codebook = codebook.to(flat.device)

    norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    unit  = flat / norms
    rotated = unit @ rotation.T

    mu  = rotated.mean(dim=1, keepdim=True)
    std = rotated.std(dim=1, keepdim=True).clamp(min=1e-8)
    normed = (rotated - mu) / std

    dist    = (normed.unsqueeze(2) - codebook.reshape(1, 1, -1)).abs()
    indices = dist.argmin(dim=2).to(torch.uint8)
    
    # NEW: Pack the indices before returning!
    packed_indices = pack_4bit(indices)
    
    prefix = orig_shape[:-1]
    return (
        packed_indices.reshape(*prefix, head_dim // 2), # Size is now halved!
        norms.reshape(*prefix, 1),
        mu.reshape(*prefix, 1),
        std.reshape(*prefix, 1),
    )

def decompress_tensor(packed_indices, norms, mu, std, rotation, codebook, target_dtype=torch.float16):
    # NEW: Unpack the indices first!
    indices = unpack_4bit(packed_indices)
    
    orig_shape = indices.shape
    head_dim = orig_shape[-1]
    flat_idx   = indices.reshape(-1, head_dim).long()
    flat_norms = norms.reshape(-1, 1).float()
    flat_mu    = mu.reshape(-1, 1).float()
    flat_std   = std.reshape(-1, 1).float()

    rotation = rotation.to(flat_idx.device)
    codebook = codebook.to(flat_idx.device)

    looked_up = codebook[flat_idx]
    unscaled  = looked_up * flat_std + flat_mu
    unrotated = unscaled @ rotation
    restored  = unrotated * flat_norms

    return restored.reshape(orig_shape).to(target_dtype)

# ═════════════════════════════════════════════════════════════════════════════
# Cache Class & Injector
# ═════════════════════════════════════════════════════════════════════════════

class HybridTurboQuantCache:
    def __init__(self, real_cache, bits: int = 4, residual_len: int = 64):
        self._real   = real_cache
        self.bits    = bits
        self.residual_len = residual_len
        self.rotation = make_rotation_matrix(HEAD_DIM)
        self.codebook = make_codebook(bits)
        self._compressed = {}
        self.n_compressed = 0
        self.n_passthrough = 0

    def has_previous_state(self, layer_idx: int = None) -> bool:
        return self._real.has_previous_state(layer_idx)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx not in ATTENTION_LAYER_INDICES:
            self.n_passthrough += 1
            return self._real.update(key_states, value_states, layer_idx, cache_kwargs)

        self.n_compressed += 1

        if layer_idx not in self._compressed:
            self._compressed[layer_idx] = {
                "k_residual": key_states, "v_residual": value_states,
                "k_idx": None, "k_norms": None, "k_mu": None, "k_std": None,
                "v_idx": None, "v_norms": None, "v_mu": None, "v_std": None,
            }
            return key_states, value_states

        store = self._compressed[layer_idx]
        store["k_residual"] = torch.cat([store["k_residual"], key_states], dim=2)
        store["v_residual"] = torch.cat([store["v_residual"], value_states], dim=2)
        seq_len = store["k_residual"].shape[2]

        if seq_len > self.residual_len:
            n_to_compress = seq_len - self.residual_len
            k_old = store["k_residual"][:, :, :n_to_compress, :]
            v_old = store["v_residual"][:, :, :n_to_compress, :]

            k_idx, k_norms, k_mu, k_std = compress_tensor(k_old, self.rotation, self.codebook)
            v_idx, v_norms, v_mu, v_std = compress_tensor(v_old, self.rotation, self.codebook)

            store["k_idx"]   = k_idx   if store["k_idx"]   is None else torch.cat([store["k_idx"],   k_idx],   dim=2)
            store["k_norms"] = k_norms if store["k_norms"] is None else torch.cat([store["k_norms"], k_norms], dim=2)
            store["k_mu"]    = k_mu    if store["k_mu"]    is None else torch.cat([store["k_mu"],    k_mu],    dim=2)
            store["k_std"]   = k_std   if store["k_std"]   is None else torch.cat([store["k_std"],   k_std],   dim=2)
            store["v_idx"]   = v_idx   if store["v_idx"]   is None else torch.cat([store["v_idx"],   v_idx],   dim=2)
            store["v_norms"] = v_norms if store["v_norms"] is None else torch.cat([store["v_norms"], v_norms], dim=2)
            store["v_mu"]    = v_mu    if store["v_mu"]    is None else torch.cat([store["v_mu"],    v_mu],    dim=2)
            store["v_std"]   = v_std   if store["v_std"]   is None else torch.cat([store["v_std"],   v_std],   dim=2)

            store["k_residual"] = store["k_residual"][:, :, n_to_compress:, :]
            store["v_residual"] = store["v_residual"][:, :, n_to_compress:, :]

        if store["k_idx"] is not None:
            k_old_decomp = decompress_tensor(store["k_idx"], store["k_norms"], store["k_mu"], store["k_std"], self.rotation, self.codebook, target_dtype=store["k_residual"].dtype)
            v_old_decomp = decompress_tensor(store["v_idx"], store["v_norms"], store["v_mu"], store["v_std"], self.rotation, self.codebook, target_dtype=store["v_residual"].dtype)
            full_k = torch.cat([k_old_decomp, store["k_residual"]], dim=2)
            full_v = torch.cat([v_old_decomp, store["v_residual"]], dim=2)
        else:
            full_k = store["k_residual"]
            full_v = store["v_residual"]

        return full_k, full_v

    def __getattr__(self, name):
        return getattr(self._real, name)

    def memory_stats(self):
        compressed_bytes = 0
        residual_bytes   = 0
        for store in self._compressed.values():
            if store["k_idx"] is not None:
                # The numel() of idx is now half of what it used to be!
                compressed_bytes += store["k_idx"].numel()    
                compressed_bytes += store["k_norms"].numel() * 4
                compressed_bytes += store["k_mu"].numel()    * 4
                compressed_bytes += store["k_std"].numel()   * 4
                compressed_bytes += store["v_idx"].numel()
                compressed_bytes += store["v_norms"].numel() * 4
                compressed_bytes += store["v_mu"].numel()    * 4
                compressed_bytes += store["v_std"].numel()   * 4
            if store["k_residual"] is not None:
                residual_bytes += store["k_residual"].numel() * 2
                residual_bytes += store["v_residual"].numel() * 2
        return {
            "compressed_kb": compressed_bytes / 1024,
            "residual_kb":   residual_bytes   / 1024,
            "total_kb":      (compressed_bytes + residual_bytes) / 1024,
        }

def inject_turbo_cache(model, bits: int = 4, residual_len: int = 64):
    original_generate = model.generate.__func__
    def patched_generate(self_model, *args, **kwargs):
        return original_generate(self_model, *args, **kwargs)

    turbo_cache_holder = {"cache": None, "injected": False}
    def forward_hook(module, input, kwargs_hook):
        if not turbo_cache_holder["injected"] and "past_key_values" in kwargs_hook:
            real_cache = kwargs_hook["past_key_values"]
            if real_cache is not None and not isinstance(real_cache, HybridTurboQuantCache):
                wrapped = HybridTurboQuantCache(real_cache, bits=bits, residual_len=residual_len)
                kwargs_hook["past_key_values"] = wrapped
                turbo_cache_holder["cache"]    = wrapped
                turbo_cache_holder["injected"] = True
        return input, kwargs_hook

    handle = model.model.register_forward_pre_hook(forward_hook, with_kwargs=True)
    return handle, turbo_cache_holder

# ═════════════════════════════════════════════════════════════════════════════
# Test Code
# ═════════════════════════════════════════════════════════════════════════════

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

def sep(title=""):
    w = 62
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("\n" + "─" * w)

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
            return_dict_in_generate=True,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if handle is not None:
        handle.remove()

    generated = out.sequences[0][n_input:]
    text      = tokenizer.decode(generated, skip_special_tokens=True)
    tps       = len(generated) / elapsed
    peak_mb   = torch.cuda.max_memory_allocated() / 1e6

    # Measure actual cache bytes directly from the returned cache tensors
    cache_mb = None
    if cache_holder and cache_holder["cache"]:
        stats    = cache_holder["cache"].memory_stats()
        cache_mb = stats["total_kb"] / 1024
    elif out.past_key_values is not None:
        pkv = out.past_key_values
        cache_bytes = 0
        # transformers v5+ DynamicCache stores per-layer state in .layers
        layers = getattr(pkv, "layers", None)
        if layers is not None:
            for layer_cache in layers:
                for t in vars(layer_cache).values():
                    if isinstance(t, torch.Tensor):
                        cache_bytes += t.numel() * t.element_size()
        # Fallback: older API with key_cache / value_cache lists
        if cache_bytes == 0:
            for attr_name in ("key_cache", "value_cache"):
                lst = getattr(pkv, attr_name, None)
                if isinstance(lst, list):
                    for t in lst:
                        if isinstance(t, torch.Tensor):
                            cache_bytes += t.numel() * t.element_size()
        cache_mb = cache_bytes / (1024 * 1024)

    result = {
        "label":    label,
        "text":     text,
        "tps":      tps,
        "peak_mb":  peak_mb,
        "n_input":  n_input,
        "tokens":   len(generated),
        "cache_mb": cache_mb,
    }
    return result

def main():
    sep("Step 4 — Bit-Packed Hybrid Cache")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=BNB_CONFIG, device_map={"": 0}, dtype=torch.float16
    ).eval()

    configs = [("Baseline (no TQ)", None), ("TurboQuant 4-bit PACKED", 4)]
    all_results = {cfg: [] for cfg, _ in configs}

    for cfg_label, bits in configs:
        sep(cfg_label)
        for i, prompt in enumerate(TEST_PROMPTS):
            p_label = ["short", "long-context"][i]
            r = run_test(model, tokenizer, prompt, cfg_label, bits=bits)
            all_results[cfg_label].append(r)
            print(f"  [{p_label}] → {r['tps']:.1f} tok/s | peak {r['peak_mb']:.0f} MB")
            if r["cache_mb"] is not None:
                label_suffix = "(vs fp16 baseline)" if bits is not None else "(FP16, measured)"
                print(f"      Cache Size: {r['cache_mb']:.2f} MB {label_suffix}")
                
    sep("SUMMARY")
    for cfg_label, results in all_results.items():
        print(f"\n{cfg_label}:")
        for i, r in enumerate(results):
            p = ["short", "long "][i]
            print(f"  {p}: {r['cache_mb']:.2f} MB Cache size")

if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# Results: it worked!
# -----------------------------------------------------------------------------
# ❯ python .\step-03-cache-updated.py

# ────────────── Step 4 — Bit-Packed Hybrid Cache ──────────────
# W0402 17:51:56.132000 24740 site-packages\torch\utils\flop_counter.py:29] triton not found; flop counting will not work for triton kernels
# Fetching 4 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 4003.15it/s]
# Download complete: : 0.00B [00:00, ?B/s]              The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
# Download complete: : 0.00B [00:00, ?B/s]
# Loading weights:   0%|▊                                                                                                                                                                                      | 2/427 [00:00<03:26,  2.06it/s]D:\miniconda3\envs\turboquant_env\Lib\site-packages\bitsandbytes\backends\cuda\ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
#   torch._check_is_size(blocksize)
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 427/427 [00:04<00:00, 91.56it/s]

# ────────────────────── Baseline (no TQ) ──────────────────────
# Setting `pad_token_id` to `eos_token_id`:248044 for open-end generation.
# D:\miniconda3\envs\turboquant_env\Lib\site-packages\bitsandbytes\backends\cuda\ops.py:468: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
#   torch._check_is_size(blocksize)
#   [short] → 12.0 tok/s | peak 7886 MB
# Setting `pad_token_id` to `eos_token_id`:248044 for open-end generation.
#   [long-context] → 9.5 tok/s | peak 11638 MB

# ────────────────── TurboQuant 4-bit PACKED ───────────────────
# Setting `pad_token_id` to `eos_token_id`:248044 for open-end generation.
#   [short] → 11.2 tok/s | peak 7886 MB
#       Cache Size: 2.83 MB (vs fp16 baseline)
# Setting `pad_token_id` to `eos_token_id`:248044 for open-end generation.
#   [long-context] → 7.9 tok/s | peak 11638 MB
#       Cache Size: 42.81 MB (vs fp16 baseline)

# ────────────────────────── SUMMARY ───────────────────────────

# Baseline (no TQ):
#   short: ~5.06 MB Cache size (FP16 Estimate)
#   long : ~151.28 MB Cache size (FP16 Estimate)

# TurboQuant 4-bit PACKED:
#   short: 2.83 MB Cache size
#   long : 42.81 MB Cache size

# --------------------------------------------------------------
# After updating to measure actual cache size directly from tensors:
# ───────────────────────── SUMMARY ────────────────────────
# **Test Configuration:** Qwen/Qwen3.5-9B (4,691 input tokens, measured directly from the cache tensors)

# | Configuration               | Tokens/sec | Cache VRAM (Measured) |
# | :-------------------------- | :--------- | :-------------------- |
# | **Baseline (FP16)**         | 9.5 tok/s  | 176.75 MB             |
# | **TurboQuant 4-Bit PACKED** | 7.9 tok/s  | **42.81 MB**          |
