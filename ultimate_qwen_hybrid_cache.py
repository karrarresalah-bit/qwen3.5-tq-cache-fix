# ultimate_qwen_hybrid_cache_v2.py
"""
Ultimate Hybrid Cache V2 — Fully Wired Edition
===============================================
Combines:
  - PATH A: Attention layers [3,7,11,15,19,23,27,31]
            → TurboQuant 4-bit bit-packed KV compression
            → residual window keeps last N tokens hot in fp16
  - PATH B: DeltaNet layers (all others)
            → FLAP variance-based importance pruning
            → position-aware sparsity (early=40%, late=55%)
            → stored as PyTorch sparse COO tensors

Fixes vs the truncated version:
  [x] Attention PATH A is fully wired (not just passthrough)
  [x] memory_stats() counts BOTH attention and delta storage
  [x] benchmark uses stats["total_mb"] not just delta_vram_mb
  [x] debug prints cleaned up (single injection message kept)
"""

import torch
from transformers import BitsAndBytesConfig

try:
    from transformers.cache_utils import Cache
    BaseCache = Cache
except ImportError:
    BaseCache = object

# ── Constants (from step 1 inspection) ───────────────────────────────────────
ATTENTION_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
HEAD_DIM         = 256

# Position-aware sparsity targets (from step 5 calibration)
EARLY_LAYER_LIMIT = 11
EARLY_SPARSITY    = 0.40   # cosine 0.9982 ✅
LATE_SPARSITY     = 0.55   # cosine 0.9903 ✅


# ═════════════════════════════════════════════════════════════════════════════
# TurboQuant helpers (attention compression)
# ═════════════════════════════════════════════════════════════════════════════

def _make_rotation(dim: int = HEAD_DIM, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(dim, dim))
    return Q

def _make_codebook(bits: int = 4) -> torch.Tensor:
    return torch.linspace(-3.0, 3.0, 2 ** bits)

# Build once at import time — reused for every compress/decompress call
_ROTATION = _make_rotation()
_CODEBOOK = _make_codebook(4)


def _compress(x: torch.Tensor, rotation: torch.Tensor, codebook: torch.Tensor):
    """
    Vectorized TurboQuant compress for a KV tensor of any shape [..., HEAD_DIM].
    Returns (packed_uint8, norms, mu, std) — all needed for decompression.
    """
    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1]).float()
    N    = flat.shape[0]

    rotation = rotation.to(flat.device)
    codebook = codebook.to(flat.device)

    # Normalize each vector to unit length
    norms  = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)   # [N, 1]
    unit   = flat / norms                                       # [N, HEAD_DIM]

    # Rotate — spreads outlier energy evenly
    rotated = unit @ rotation.T                                 # [N, HEAD_DIM]

    # Normalize to codebook range
    mu  = rotated.mean(dim=1, keepdim=True)                    # [N, 1]
    std = rotated.std(dim=1,  keepdim=True).clamp(min=1e-8)    # [N, 1]
    normed = (rotated - mu) / std                              # [N, HEAD_DIM]

    # Quantize — nearest codebook entry for every value
    dist    = (normed.unsqueeze(2) - codebook.reshape(1, 1, -1)).abs()
    indices = dist.argmin(dim=2).to(torch.uint8)               # [N, HEAD_DIM]

    # Bit-pack: two 4-bit values → one uint8 byte  (saves 2×)
    even   = indices[:, 0::2]                                  # [N, HEAD_DIM//2]
    odd    = indices[:, 1::2]
    packed = (even & 0x0F) | ((odd & 0x0F) << 4)              # [N, HEAD_DIM//2]

    prefix = orig_shape[:-1]
    return (
        packed.reshape(*prefix, HEAD_DIM // 2),
        norms.reshape(*prefix, 1),
        mu.reshape(*prefix, 1),
        std.reshape(*prefix, 1),
    )


def _decompress(packed, norms, mu, std,
                rotation: torch.Tensor,
                codebook: torch.Tensor,
                dtype=torch.float16) -> torch.Tensor:
    """Decompress back to a KV tensor."""
    orig_shape = packed.shape[:-1] + (HEAD_DIM,)
    flat_pk = packed.reshape(-1, HEAD_DIM // 2)
    flat_no = norms.reshape(-1, 1).float()
    flat_mu = mu.reshape(-1, 1).float()
    flat_st = std.reshape(-1, 1).float()
    N = flat_pk.shape[0]

    rotation = rotation.to(flat_pk.device)
    codebook = codebook.to(flat_pk.device)

    # Unpack bits
    even = flat_pk & 0x0F                               # low nibble
    odd  = (flat_pk >> 4) & 0x0F                        # high nibble
    indices = torch.empty(N, HEAD_DIM, dtype=torch.long, device=flat_pk.device)
    indices[:, 0::2] = even.long()
    indices[:, 1::2] = odd.long()

    # Look up codebook → undo normalization → rotate back → restore norm
    looked   = codebook[indices]                         # [N, HEAD_DIM]
    unscaled = looked * flat_st + flat_mu               # undo range norm
    unrot    = unscaled @ rotation                       # rotate backwards (R.T.T = R)
    restored = unrot * flat_no                           # restore original scale

    return restored.reshape(orig_shape).to(dtype)


# ═════════════════════════════════════════════════════════════════════════════
# FLAP pruning helper (DeltaNet compression)
# ═════════════════════════════════════════════════════════════════════════════

def _flap_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Variance-based importance pruning.
    Low-variance features (unimportant) → zeroed out → stored sparse.
    """
    if sparsity <= 0.0:
        return tensor.to_sparse()

    # Importance = variance along last dim, broadcast back
    importance = tensor.var(dim=-1, keepdim=True)
    threshold  = torch.quantile(importance.float(), sparsity).to(tensor.dtype)
    mask       = (importance >= threshold)
    pruned     = tensor * mask

    return pruned.to_sparse()


def _sparse_bytes(t: torch.Tensor) -> int:
    """Bytes used by a sparse COO tensor (values + indices)."""
    if not t.is_sparse:
        return t.numel() * t.element_size()
    return (
        t._values().numel() * t._values().element_size()
        + t._indices().numel() * t._indices().element_size()
    )


# ═════════════════════════════════════════════════════════════════════════════
# The cache
# ═════════════════════════════════════════════════════════════════════════════

class UltimateHybridCacheV2(BaseCache):
    """
    Drop-in cache replacement for Qwen3.5-9B that compresses:
      - Attention KV pairs  → 4-bit TurboQuant bit-packed
      - DeltaNet states     → FLAP variance-based sparse pruning

    Injected via forward pre-hook — zero model weight changes.
    """

    def __init__(self, real_cache, bits: int = 4, residual_len: int = 64):
        self._real        = real_cache
        self.bits         = bits
        self.residual_len = residual_len

        # Attention storage per layer:
        # { layer_idx: { k_packed, k_norms, k_mu, k_std,
        #                v_packed, v_norms, v_mu, v_std,
        #                k_residual, v_residual } }
        self._attn = {}

        # DeltaNet storage per layer:
        # { layer_idx: (k_sparse, v_sparse) }
        self._delta = {}

        # Lazy-move rotation/codebook to GPU on first use
        self._rotation = _ROTATION
        self._codebook = _CODEBOOK

    # ── Delegate everything except our overrides to the real cache ────────────
    def __getattribute__(self, name):
        _ours = {
            "__init__", "_real", "bits", "residual_len",
            "_attn", "_delta", "_rotation", "_codebook",
            "__class__", "update", "memory_stats",
            "_update_attention", "_update_delta",
        }
        if name in _ours:
            return object.__getattribute__(self, name)
        return getattr(object.__getattribute__(self, "_real"), name)

    # ── Main entry point called by every layer ────────────────────────────────
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):

        # ── PATH A: Attention layer → TurboQuant ─────────────────────────────
        if layer_idx in ATTENTION_LAYERS:
            return self._update_attention(key_states, value_states, layer_idx)

        # ── PATH B: DeltaNet layer → FLAP sparse pruning ─────────────────────
        return self._update_delta(key_states, value_states, layer_idx, cache_kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    def _update_attention(self, key_states, value_states, layer_idx):
        """
        TurboQuant 4-bit compression for classic attention KV pairs.
        Keeps the most recent `residual_len` tokens hot in fp16,
        compresses everything older into bit-packed uint8 indices.
        """
        device = key_states.device

        # Move rotation/codebook to GPU lazily
        if self._rotation.device != device:
            self._rotation = self._rotation.to(device)
            self._codebook = self._codebook.to(device)

        if layer_idx not in self._attn:
            # First call — just store as residual, nothing to compress yet
            self._attn[layer_idx] = {
                "k_residual": key_states,
                "v_residual": value_states,
                "k_packed": None, "k_norms": None,
                "k_mu":     None, "k_std":   None,
                "v_packed": None, "v_norms": None,
                "v_mu":     None, "v_std":   None,
            }
            return key_states, value_states

        store = self._attn[layer_idx]

        # Append new tokens to residual window
        store["k_residual"] = torch.cat([store["k_residual"], key_states],   dim=2)
        store["v_residual"] = torch.cat([store["v_residual"], value_states], dim=2)

        # If residual overflows → compress the old part
        if store["k_residual"].shape[2] > self.residual_len:
            n_old = store["k_residual"].shape[2] - self.residual_len

            k_old = store["k_residual"][:, :, :n_old, :]
            v_old = store["v_residual"][:, :, :n_old, :]

            kp, kn, km, ks = _compress(k_old, self._rotation, self._codebook)
            vp, vn, vm, vs = _compress(v_old, self._rotation, self._codebook)

            # Accumulate compressed chunks
            def _cat(existing, new):
                return new if existing is None else torch.cat([existing, new], dim=2)

            store["k_packed"] = _cat(store["k_packed"], kp)
            store["k_norms"]  = _cat(store["k_norms"],  kn)
            store["k_mu"]     = _cat(store["k_mu"],     km)
            store["k_std"]    = _cat(store["k_std"],     ks)
            store["v_packed"] = _cat(store["v_packed"], vp)
            store["v_norms"]  = _cat(store["v_norms"],  vn)
            store["v_mu"]     = _cat(store["v_mu"],     vm)
            store["v_std"]    = _cat(store["v_std"],     vs)

            # Trim residual
            store["k_residual"] = store["k_residual"][:, :, n_old:, :]
            store["v_residual"] = store["v_residual"][:, :, n_old:, :]

        # Build full KV for attention: decompress old + concat hot residual
        if store["k_packed"] is not None:
            k_old = _decompress(store["k_packed"], store["k_norms"],
                                store["k_mu"],     store["k_std"],
                                self._rotation,    self._codebook,
                                dtype=store["k_residual"].dtype)
            v_old = _decompress(store["v_packed"], store["v_norms"],
                                store["v_mu"],     store["v_std"],
                                self._rotation,    self._codebook,
                                dtype=store["v_residual"].dtype)
            full_k = torch.cat([k_old, store["k_residual"]], dim=2)
            full_v = torch.cat([v_old, store["v_residual"]], dim=2)
        else:
            full_k = store["k_residual"]
            full_v = store["v_residual"]

        return full_k, full_v

    # ─────────────────────────────────────────────────────────────────────────
    def _update_delta(self, key_states, value_states, layer_idx, cache_kwargs):
        """
        FLAP variance-based pruning for DeltaNet recurrent state tensors.
        Restores sparse state before update, then re-prunes after.
        """
        sparsity = EARLY_SPARSITY if layer_idx <= EARLY_LAYER_LIMIT else LATE_SPARSITY

        # Restore previously stored sparse state into the real cache
        if layer_idx in self._delta:
            k_sp, v_sp = self._delta[layer_idx]
            try:
                self._real.key_cache[layer_idx]   = k_sp.to_dense()
                self._real.value_cache[layer_idx] = v_sp.to_dense()
            except (AttributeError, IndexError):
                pass  # some DeltaNet layers don't use key/value cache slots

        # Let the real cache do its normal update
        new_k, new_v = self._real.update(key_states, value_states, layer_idx, cache_kwargs)

        # Prune and store sparse
        self._delta[layer_idx] = (
            _flap_prune(new_k, sparsity),
            _flap_prune(new_v, sparsity),
        )

        return new_k, new_v

    # ─────────────────────────────────────────────────────────────────────────
    def memory_stats(self) -> dict:
        """
        Measure VRAM used by our compressed storage.
        Returns bytes for attention (bit-packed) and delta (sparse) separately.
        """
        attn_bytes = 0
        for store in self._attn.values():
            for key in ("k_packed", "k_norms", "k_mu", "k_std",
                        "v_packed", "v_norms", "v_mu", "v_std"):
                t = store.get(key)
                if t is not None:
                    attn_bytes += t.numel() * t.element_size()
            # residual windows (fp16)
            for key in ("k_residual", "v_residual"):
                t = store.get(key)
                if t is not None:
                    attn_bytes += t.numel() * t.element_size()

        delta_bytes = 0
        for k_sp, v_sp in self._delta.values():
            delta_bytes += _sparse_bytes(k_sp)
            delta_bytes += _sparse_bytes(v_sp)

        total_bytes = attn_bytes + delta_bytes

        return {
            "attn_mb":  attn_bytes  / (1024 ** 2),
            "delta_mb": delta_bytes / (1024 ** 2),
            "total_mb": total_bytes / (1024 ** 2),
            # keep old key so benchmark_v2.py still works
            "delta_vram_mb": total_bytes / (1024 ** 2),
            "attn_layers_compressed":  len(self._attn),
            "delta_layers_compressed": len(self._delta),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Injection helper
# ═════════════════════════════════════════════════════════════════════════════

def inject_ultimate_cache(model, bits: int = 4, residual_len: int = 64):
    """
    Registers a forward pre-hook that swaps in UltimateHybridCacheV2
    the moment Qwen3.5 builds its real cache during generation.
    """
    cache_holder = {"cache": None, "injected": False}

    def forward_hook(module, args, kwargs_hook):
        if not cache_holder["injected"] and "past_key_values" in kwargs_hook:
            real_cache = kwargs_hook["past_key_values"]
            if real_cache is not None and not isinstance(real_cache, UltimateHybridCacheV2):
                wrapped = UltimateHybridCacheV2(
                    real_cache, bits=bits, residual_len=residual_len
                )
                kwargs_hook["past_key_values"] = wrapped
                cache_holder["cache"]    = wrapped
                cache_holder["injected"] = True
                print(f"  ✅ UltimateHybridCacheV2 injected!  "
                      f"(bits={bits}, residual_len={residual_len})")
        return args, kwargs_hook

    handle = model.model.register_forward_pre_hook(forward_hook, with_kwargs=True)
    return handle, cache_holder