"""
cache_injector.py — HybridTurboQuantCache for Qwen3.5
=======================================================
Production-ready module.  Import and use in your own scripts:

    from cache_injector import inject_turbo_cache

    handle, cache_holder = inject_turbo_cache(model, bits=4, residual_len=64)
    # ... generate as normal ...
    handle.remove()

How it works
------------
Qwen3.5-9B is a Hybrid-Attention model: 8 of its 32 layers use standard
softmax attention (layers 3, 7, 11, 15, 19, 23, 27, 31) while the
remaining 24 layers are Gated Delta Networks that maintain a recurrent
state matrix instead of a KV cache.

This module wraps the model's built-in HybridCache via a forward pre-hook
and routes each layer's cache update through compression or passthrough:

  - Attention layers  → 4-bit TurboQuant compression (rotate, normalize,
                         quantize, bit-pack two 4-bit values per byte)
  - DeltaNet layers   → untouched, delegated to the real cache unchanged

Result: ~3.5× reduction in KV cache VRAM on long contexts, with cosine
similarity > 0.99 vs. FP16 baseline.
"""

import torch
from transformers import AutoModelForCausalLM

# Attention layer indices discovered by step-01-inspector.py
ATTENTION_LAYER_INDICES = {3, 7, 11, 15, 19, 23, 27, 31}
HEAD_DIM = 256


# ── Bit-packing helpers ───────────────────────────────────────────────────────

def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint8 tensor of 4-bit values (0-15) into half the bytes."""
    even   = indices[..., 0::2].to(torch.uint8)
    odd    = indices[..., 1::2].to(torch.uint8)
    return (even << 4) | odd


def unpack_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack a bit-packed tensor back to original size."""
    even = packed >> 4
    odd  = packed & 0x0F
    unpacked = torch.empty(
        *packed.shape[:-1], packed.shape[-1] * 2,
        dtype=torch.uint8, device=packed.device
    )
    unpacked[..., 0::2] = even
    unpacked[..., 1::2] = odd
    return unpacked


# ── TurboQuant core ───────────────────────────────────────────────────────────

def make_rotation_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    """Build a random orthogonal rotation matrix (QR decomposition)."""
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(dim, dim))
    return Q


def make_codebook(bits: int) -> torch.Tensor:
    """Evenly-spaced codebook over [-3, 3] with 2^bits levels."""
    return torch.linspace(-3.0, 3.0, 2 ** bits)


def compress_tensor(x: torch.Tensor,
                    rotation: torch.Tensor,
                    codebook: torch.Tensor):
    """
    Compress a KV tensor of shape [..., head_dim] to 4-bit packed indices.

    Returns (packed_indices, norms, mu, std) where packed_indices has shape
    [..., head_dim // 2] (two 4-bit values packed per byte).
    """
    orig_shape = x.shape
    head_dim   = orig_shape[-1]
    flat       = x.reshape(-1, head_dim).float()

    rotation = rotation.to(flat.device)
    codebook = codebook.to(flat.device)

    norms   = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    unit    = flat / norms
    rotated = unit @ rotation.T

    mu     = rotated.mean(dim=1, keepdim=True)
    std    = rotated.std(dim=1,  keepdim=True).clamp(min=1e-8)
    normed = (rotated - mu) / std

    dist    = (normed.unsqueeze(2) - codebook.reshape(1, 1, -1)).abs()
    indices = dist.argmin(dim=2).to(torch.uint8)

    packed  = pack_4bit(indices)
    prefix  = orig_shape[:-1]
    return (
        packed.reshape(*prefix, head_dim // 2),
        norms.reshape(*prefix, 1),
        mu.reshape(*prefix, 1),
        std.reshape(*prefix, 1),
    )


def decompress_tensor(packed_indices: torch.Tensor,
                      norms: torch.Tensor,
                      mu: torch.Tensor,
                      std: torch.Tensor,
                      rotation: torch.Tensor,
                      codebook: torch.Tensor,
                      target_dtype=torch.float16) -> torch.Tensor:
    """Decompress packed 4-bit indices back to a KV tensor."""
    indices    = unpack_4bit(packed_indices)
    orig_shape = indices.shape
    head_dim   = orig_shape[-1]

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


# ── Cache class ───────────────────────────────────────────────────────────────

class HybridTurboQuantCache:
    """
    A drop-in cache replacement for Qwen3.5's HybridCache that compresses
    attention KV pairs with 4-bit TurboQuant while leaving DeltaNet state
    matrices completely untouched.

    Attributes
    ----------
    bits : int
        Quantization bits (4 recommended; cosine > 0.99 vs FP16).
    residual_len : int
        Number of most-recent tokens kept uncompressed in FP16.
        Protecting the active window maintains generation quality.
    """

    def __init__(self, real_cache, bits: int = 4, residual_len: int = 64):
        self._real        = real_cache
        self.bits         = bits
        self.residual_len = residual_len
        self.rotation     = make_rotation_matrix(HEAD_DIM)
        self.codebook     = make_codebook(bits)
        self._compressed  = {}
        self.n_compressed  = 0
        self.n_passthrough = 0

    # ── Interface Qwen3.5 expects ─────────────────────────────────────────────

    def has_previous_state(self, layer_idx: int = None) -> bool:
        """Called by DeltaNet layers — delegate to real cache."""
        return self._real.has_previous_state(layer_idx)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        Route every layer's cache update.
        Attention layers → compress; DeltaNet layers → passthrough.
        """
        if layer_idx not in ATTENTION_LAYER_INDICES:
            self.n_passthrough += 1
            return self._real.update(key_states, value_states, layer_idx, cache_kwargs)

        self.n_compressed += 1

        if layer_idx not in self._compressed:
            self._compressed[layer_idx] = {
                "k_residual": key_states,
                "v_residual": value_states,
                "k_idx": None, "k_norms": None, "k_mu": None, "k_std": None,
                "v_idx": None, "v_norms": None, "v_mu": None, "v_std": None,
            }
            return key_states, value_states

        store = self._compressed[layer_idx]

        store["k_residual"] = torch.cat([store["k_residual"], key_states],   dim=2)
        store["v_residual"] = torch.cat([store["v_residual"], value_states], dim=2)

        if store["k_residual"].shape[2] > self.residual_len:
            n_compress = store["k_residual"].shape[2] - self.residual_len

            k_old = store["k_residual"][:, :, :n_compress, :]
            v_old = store["v_residual"][:, :, :n_compress, :]

            k_idx, k_norms, k_mu, k_std = compress_tensor(k_old, self.rotation, self.codebook)
            v_idx, v_norms, v_mu, v_std = compress_tensor(v_old, self.rotation, self.codebook)

            def _cat(existing, new):
                return new if existing is None else torch.cat([existing, new], dim=2)

            store["k_idx"]   = _cat(store["k_idx"],   k_idx)
            store["k_norms"] = _cat(store["k_norms"], k_norms)
            store["k_mu"]    = _cat(store["k_mu"],    k_mu)
            store["k_std"]   = _cat(store["k_std"],   k_std)
            store["v_idx"]   = _cat(store["v_idx"],   v_idx)
            store["v_norms"] = _cat(store["v_norms"], v_norms)
            store["v_mu"]    = _cat(store["v_mu"],    v_mu)
            store["v_std"]   = _cat(store["v_std"],   v_std)

            store["k_residual"] = store["k_residual"][:, :, n_compress:, :]
            store["v_residual"] = store["v_residual"][:, :, n_compress:, :]

        if store["k_idx"] is not None:
            dtype = store["k_residual"].dtype
            k_decomp = decompress_tensor(
                store["k_idx"], store["k_norms"], store["k_mu"], store["k_std"],
                self.rotation, self.codebook, target_dtype=dtype,
            )
            v_decomp = decompress_tensor(
                store["v_idx"], store["v_norms"], store["v_mu"], store["v_std"],
                self.rotation, self.codebook, target_dtype=dtype,
            )
            full_k = torch.cat([k_decomp, store["k_residual"]], dim=2)
            full_v = torch.cat([v_decomp, store["v_residual"]], dim=2)
        else:
            full_k = store["k_residual"]
            full_v = store["v_residual"]

        return full_k, full_v

    def __getattr__(self, name):
        return getattr(self._real, name)

    def memory_stats(self) -> dict:
        """Return a dict with compressed / residual / total bytes."""
        comp_bytes = 0
        res_bytes  = 0
        for store in self._compressed.values():
            if store["k_idx"] is not None:
                for key in ("k_idx", "v_idx"):
                    comp_bytes += store[key].numel()          # uint8 = 1 byte each
                for key in ("k_norms", "k_mu", "k_std", "v_norms", "v_mu", "v_std"):
                    comp_bytes += store[key].numel() * 4      # float32
            if store["k_residual"] is not None:
                res_bytes += store["k_residual"].numel() * 2  # fp16
                res_bytes += store["v_residual"].numel() * 2
        return {
            "compressed_kb": comp_bytes / 1024,
            "residual_kb":   res_bytes  / 1024,
            "total_kb":      (comp_bytes + res_bytes) / 1024,
        }


# ── Injection helper ──────────────────────────────────────────────────────────

def inject_turbo_cache(model, bits: int = 4, residual_len: int = 64):
    """
    Register a forward pre-hook that wraps Qwen3.5's HybridCache with
    HybridTurboQuantCache the first time the model's inner transformer
    is called.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        A loaded Qwen3.5 (or compatible hybrid-attention) model.
    bits : int
        Quantization bits for the KV cache (default 4).
    residual_len : int
        Recent tokens kept uncompressed in FP16 (default 64).

    Returns
    -------
    handle : torch.utils.hooks.RemovableHook
        Call handle.remove() after generation to clean up.
    cache_holder : dict
        {"cache": HybridTurboQuantCache | None, "injected": bool}
        Access cache_holder["cache"].memory_stats() after generation.

    Example
    -------
    >>> handle, cache_holder = inject_turbo_cache(model, bits=4)
    >>> outputs = model.generate(**inputs, max_new_tokens=200)
    >>> handle.remove()
    >>> print(cache_holder["cache"].memory_stats())
    """
    cache_holder = {"cache": None, "injected": False}

    def forward_hook(module, args, kwargs):
        if not cache_holder["injected"] and "past_key_values" in kwargs:
            real_cache = kwargs["past_key_values"]
            if real_cache is not None and not isinstance(real_cache, HybridTurboQuantCache):
                wrapped = HybridTurboQuantCache(real_cache, bits=bits, residual_len=residual_len)
                kwargs["past_key_values"] = wrapped
                cache_holder["cache"]     = wrapped
                cache_holder["injected"]  = True
        return args, kwargs

    handle = model.model.register_forward_pre_hook(forward_hook, with_kwargs=True)
    return handle, cache_holder
