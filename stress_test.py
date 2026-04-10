"""
stress_test_v2.py
====================================================
Stress Testing the Ultimate Hybrid Cache V2 
Pushing the context length and generation limits!
"""

import torch
import time
import gc
import sys
sys.stdout.reconfigure(encoding='utf-8')
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Import our v2 Surgeon's edition
from ultimate_qwen_hybrid_cache import inject_ultimate_cache

def count_tensor_bytes(obj, seen=None):
    if seen is None: seen = set()
    if id(obj) in seen: return 0
    seen.add(id(obj))
    total_bytes = 0
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    elif isinstance(obj, (list, tuple)):
        for item in obj: total_bytes += count_tensor_bytes(item, seen)
    elif hasattr(obj, "__dict__"):
        for val in vars(obj).values(): total_bytes += count_tensor_bytes(val, seen)
    return total_bytes

def sep(title=""):
    w = 70
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "━" * pad + f" {title} " + "━" * (w - pad - len(title) - 2))
    else:
        print("\n" + "━" * w)

def main():
    sep("Waking up Qwen3.5 (9B) - V2 STRESS TEST")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("  ⏳ Loading tokenizer and model... (Hold onto your hats)")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-9B",
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    # The Stress Prompt: Multiply by a large number to force massive context
    base_text = (
        "The ancient city of Aethoria stood at the edge of the Silver Sea. "
        "Its towers of enchanted stone hummed softly in the ocean wind. "
        "For centuries the scholars of Aethoria devoted their lives to unraveling the mysteries of time. "
    )
    
    # 60 multiplier builds a massive context ~3000 tokens
    prompt = base_text * 60 + "\n\nAnalyze the world-building themes in the text above and write a 500-word essay about the time magic used in Aethoria, detailing 3 unique spells they might have invented."
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    gen_target = 512  # We want a very long response!
    
    print(f"  ✅ Model loaded. MEGA Prompt length: {input_length} tokens.")
    print(f"  🎯 Target Generation: {gen_target} tokens.")

    # ══════════════════════════════════════════════════════════════════════════
    # RUN 1: BASELINE
    # ══════════════════════════════════════════════════════════════════════════
    sep("RUN 1: Baseline (Standard FP16)")
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    try:
        with torch.no_grad():
            out_base = model.generate(
                **inputs,
                max_new_tokens=gen_target,
                do_sample=False,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )

        torch.cuda.synchronize()
        time_base = time.perf_counter() - t0
        tok_base = len(out_base.sequences[0]) - input_length
        tps_base = tok_base / time_base
        base_bytes = count_tensor_bytes(out_base.past_key_values)
        base_mb = base_bytes / (1024 * 1024)
        peak_vram_base = torch.cuda.max_memory_allocated() / (1024**3)
        text_base = tokenizer.decode(out_base.sequences[0][input_length:], skip_special_tokens=True)

        print(f"  → Speed: {tps_base:.1f} tok/s | Cache: {base_mb:.2f} MB | Peak VRAM: {peak_vram_base:.2f} GB")

        del out_base
    except torch.cuda.OutOfMemoryError:
         print("  ❌ CUDA OOM on Baseline! The cache was too huge.")
         base_mb = -1
         text_base = "OOM FAILED"

    gc.collect()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # RUN 2: ULTIMATE HYBRID CACHE V2 (FLAP + POSITION-AWARE)
    # ══════════════════════════════════════════════════════════════════════════
    sep("RUN 2: Ultimate Hybrid Cache V2 (Position-Aware + FLAP)")
    
    handle, cache_holder = inject_ultimate_cache(model, bits=4, residual_len=64)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    try:
        with torch.no_grad():
            from transformers.cache_utils import DynamicCache
            # We are ready!
            out_ult = model.generate(
                **inputs,
                max_new_tokens=gen_target,
                do_sample=False,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )

        torch.cuda.synchronize()
        time_ult = time.perf_counter() - t0
        tok_ult = len(out_ult.sequences[0]) - input_length
        tps_ult = tok_ult / time_ult
        stats = cache_holder["cache"].memory_stats()
        ult_mb = stats["total_mb"]
        peak_vram_ult = torch.cuda.max_memory_allocated() / (1024**3)
        text_ult = tokenizer.decode(out_ult.sequences[0][input_length:], skip_special_tokens=True)

        print(f"  → Speed: {tps_ult:.1f} tok/s | Cache: {ult_mb:.2f} MB | Peak VRAM: {peak_vram_ult:.2f} GB")
    except torch.cuda.OutOfMemoryError:
        print("  ❌ CUDA OOM on Ultimate V2!")
        ult_mb = -1
        text_ult = "OOM FAILED"
        
    handle.remove()

    # ══════════════════════════════════════════════════════════════════════════
    # VERDICT
    # ══════════════════════════════════════════════════════════════════════════
    ratio = (base_mb / ult_mb) if (base_mb > 0 and ult_mb > 0) else 0
    sep("🏆 EXTREME STRESS VERDICT 🏆")
    if base_mb > 0:
        print(f"  Baseline Cache Size: {base_mb:>8.2f} MB")
    else:
        print(f"  Baseline Cache Size: OOM FAILED")
        
    if ult_mb > 0:
        print(f"  Ultimate V2 Size:     {ult_mb:>8.2f} MB")
    else:
        print(f"  Ultimate V2 Size:     OOM FAILED")
        
    if ratio > 0:
        print(f"  Compression Ratio:   {ratio:>8.2f}x Smaller! 💎")
        
    print(f"\n  Quality Persistence (0-150 chars):")
    print(f"  Base: {text_base[:150].replace(chr(10), ' ')}...")
    print(f"  Ult2: {text_ult[:150].replace(chr(10), ' ')}...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

if __name__ == "__main__":
    main()
