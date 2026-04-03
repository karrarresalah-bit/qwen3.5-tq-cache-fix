import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import your custom injector!
from ultimate_qwen_hybrid_cache import inject_ultimate_cache

def main():
    print("🤖 Waking up Qwen3.5...")
    
    # 1. Standard 4-bit quantization config for the model weights
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # 2. Load the Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-9B",
        quantization_config=bnb_config,
        device_map="auto", # Let PyTorch map it to your grandma GPU!
    )
    model.eval()

    # 3. Prepare your prompt (Let's test it with something you like!)
    prompt = "Write a highly detailed explanation of how natural language processing models translate complex Japanese grammar into English."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("✨ Injecting the Ultimate Hybrid Cache...")
    # 4. INJECT THE MAGIC! 
    # bits=4 applies the TurboQuant packing to Attention
    # residual_len=64 keeps the newest tokens uncompressed for speed
    handle, cache_holder = inject_ultimate_cache(model, bits=4, residual_len=64)

    print("🚀 Generating (Watch the VRAM stay completely flat!)...")
    # 5. Generate your response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
        )

    # 6. VERY IMPORTANT: Remove the surgical hook when done!
    handle.remove()

    # 7. Decode and print the result
    input_length = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    print("\n" + "="*50)
    print("🎯 QWEN SAYS:")
    print("="*50)
    print(generated_text)
    
    # Show off the memory savings
    stats = cache_holder["cache"].memory_stats()
    print("\n" + "="*50)
    print(f"📉 FINAL CACHE SIZE: {stats['total_kb'] / 1024:.2f} MB")
    print("="*50)

if __name__ == "__main__":
    main()