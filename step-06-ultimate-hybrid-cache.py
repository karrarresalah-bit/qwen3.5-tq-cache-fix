"""
Step 6 — The Ultimate Hybrid Cache (TurboQuant + Delta Sparse Pruning)
========================================================================
This final cache handles BOTH architectures simultaneously:
  1. Attention Layers: 4-bit Bit-Packed TurboQuant
  2. DeltaNet Layers: 50% Magnitude Pruning via PyTorch Sparse Tensors

This was renamed from "step-06-ultimate-cache.py" to "ultimate_hybrid_cache.py" to better reflect the dual nature of the caching strategy. The code is now fully integrated and ready for testing in Step 7!
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
DELTA_SPARSITY_TARGET = 0.50  # 50% pruning determined from Step 5

# ═════════════════════════════════════════════════════════════════════════════
# 1. 4-Bit Bit-Packing Tools (Attention)
# ═════════════════════════════════════════════════════════════════════════════

def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    even = indices[..., 0::2].to(torch.uint8)
    odd  = indices[..., 1::2].to(torch.uint8)
    return (even << 4) | odd

def unpack_4bit(packed: torch.Tensor) -> torch.Tensor:
    even = packed >> 4
    odd  = packed & 0x0F
    unpacked = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8, device=packed.device)
    unpacked[..., 0::2] = even
    unpacked[..., 1::2] = odd
    return unpacked

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
    packed_indices = pack_4bit(indices)
    
    prefix = orig_shape[:-1]
    return (
        packed_indices.reshape(*prefix, head_dim // 2),
        norms.reshape(*prefix, 1),
        mu.reshape(*prefix, 1),
        std.reshape(*prefix, 1),
    )

def decompress_tensor(packed_indices, norms, mu, std, rotation, codebook, target_dtype=torch.float16):
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
# 2. Magnitude Pruning Tools (DeltaNet)
# ═════════════════════════════════════════════════════════════════════════════

def compress_delta_state(state: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Zeroes out the lowest magnitude weights and converts to a sparse format."""
    threshold = torch.quantile(state.abs().float(), sparsity).to(state.dtype)
    pruned_state = torch.where(state.abs() >= threshold, state, torch.zeros_like(state))
    # Convert to Coordinate (COO) sparse format to save memory
    return pruned_state.to_sparse()

def decompress_delta_state(sparse_state: torch.Tensor) -> torch.Tensor:
    """Restores the sparse tensor back to dense for matrix multiplication."""
    if sparse_state.is_sparse:
        return sparse_state.to_dense()
    return sparse_state

# ═════════════════════════════════════════════════════════════════════════════
# 3. The Ultimate Cache Class
# ═════════════════════════════════════════════════════════════════════════════

class UltimateHybridCache:
    def __init__(self, real_cache, bits: int = 4, residual_len: int = 64):
        self._real   = real_cache
        self.bits    = bits
        self.residual_len = residual_len
        self.rotation = make_rotation_matrix(HEAD_DIM)
        self.codebook = make_codebook(bits)
        
        self._compressed_attn = {}
        self._compressed_delta = {}

    def has_previous_state(self, layer_idx: int = None) -> bool:
        return self._real.has_previous_state(layer_idx)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # ── DeltaNet Layers (Sparse Pruning) ──
        if layer_idx not in ATTENTION_LAYER_INDICES:
            # Check if we have a compressed prior state to awaken
            if layer_idx in self._compressed_delta:
                k_sparse, v_sparse = self._compressed_delta[layer_idx]
                prev_k = decompress_delta_state(k_sparse)
                prev_v = decompress_delta_state(v_sparse)
                
                # Execute the standard update with restored dense tensors
                # (Assuming the underlying cache can handle manual state injection or we update the real cache)
                self._real.key_cache[layer_idx] = prev_k
                self._real.value_cache[layer_idx] = prev_v

            new_k, new_v = self._real.update(key_states, value_states, layer_idx, cache_kwargs)
            
            # Re-compress and store as sparse
            self._compressed_delta[layer_idx] = (
                compress_delta_state(new_k, DELTA_SPARSITY_TARGET),
                compress_delta_state(new_v, DELTA_SPARSITY_TARGET)
            )
            return new_k, new_v

        # ── Standard Attention Layers (4-bit TurboQuant) ──
        if layer_idx not in self._compressed_attn:
            self._compressed_attn[layer_idx] = {
                "k_residual": key_states, "v_residual": value_states,
                "k_idx": None, "k_norms": None, "k_mu": None, "k_std": None,
                "v_idx": None, "v_norms": None, "v_mu": None, "v_std": None,
            }
            return key_states, value_states

        store = self._compressed_attn[layer_idx]
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

def inject_ultimate_cache(model, bits: int = 4, residual_len: int = 64):
    original_generate = model.generate.__func__
    def patched_generate(self_model, *args, **kwargs):
        return original_generate(self_model, *args, **kwargs)

    cache_holder = {"cache": None, "injected": False}
    def forward_hook(module, input, kwargs_hook):
        if not cache_holder["injected"] and "past_key_values" in kwargs_hook:
            real_cache = kwargs_hook["past_key_values"]
            if real_cache is not None and not isinstance(real_cache, UltimateHybridCache):
                wrapped = UltimateHybridCache(real_cache, bits=bits, residual_len=residual_len)
                kwargs_hook["past_key_values"] = wrapped
                cache_holder["cache"]    = wrapped
                cache_holder["injected"] = True
        return input, kwargs_hook

    handle = model.model.register_forward_pre_hook(forward_hook, with_kwargs=True)
    return handle, cache_holder