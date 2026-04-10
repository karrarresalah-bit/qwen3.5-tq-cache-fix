from datasets import load_dataset, interleave_datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import torch.nn as nn
import numpy as np

# --- Configuration ---
MODEL_ID = "Qwen/Qwen3.5-9B"
NUM_SAMPLES = 400
SEQ_LEN = 1024
BATCH_SIZE = 1 # Keep small to avoid OOM

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# --- Dataset Preparation ---
def get_calibration_dataset(num_samples=400, seq_len=1024):
    print("🚀 Loading mixed calibration datasets (streaming mode)...")
    
    # 1. General Pre-training (Pile)
    pile = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True).shuffle(seed=42).take(240)
    pile = pile.select_columns(["text"])
    
    # 2. Reasoning (GSM8K)
    gsm = load_dataset("gsm8k", "main", split="train", streaming=True).shuffle(seed=42).take(80)
    gsm = gsm.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"})
    gsm = gsm.select_columns(["text"])
    
    # 3. Long-coherent (WikiText)
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True).shuffle(seed=42).take(80)
    wikitext = wikitext.select_columns(["text"])
    
    # Interleave and specify probabilities to avoid early exhaustion and keep proportions
    dataset = interleave_datasets(
        [pile, gsm, wikitext], 
        probabilities=[0.6, 0.2, 0.2], # Matches 240:80:80 split
        stopping_strategy="all_exhausted",
        seed=42
    )
    
    print("🧠 Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# --- FLAP Importance & Pruning Logic ---
class StateStats:
    def __init__(self):
        self.count = 0
        self.sum = None
        self.sq_sum = None

    def update(self, x):
        # x shape: [batch, ...]
        x = x.float().detach()
        if self.sum is None:
            self.sum = torch.zeros_like(x[0])
            self.sq_sum = torch.zeros_like(x[0])
        
        self.sum += x.sum(dim=0)
        self.sq_sum += (x**2).sum(dim=0)
        self.count += x.size(0)

    def get_variance(self):
        if self.count == 0: return None
        mean = self.sum / self.count
        return (self.sq_sum / self.count) - (mean**2)

def main():
    print("💎 Step 5: Advanced DeltaNet Calibration (FLAP-style)")
    
    # 1. Load Model
    print(f"  Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).eval()
    
    # 2. Setup Hooks to capture DeltaNet states
    layer_stats = {}
    layer_samples = {} # Store a sample tensor for final validation
    
    def hook_fn(module_name):
        def hook(module, input, output):
            if module_name not in layer_stats:
                layer_stats[module_name] = StateStats()
            
            # Target the Delta state tensors
            # In Qwen 3.5, recursive logic to find candidates works best
            def process_output(o):
                if isinstance(o, torch.Tensor) and o.numel() > 1000:
                    layer_stats[module_name].update(o)
                    if module_name not in layer_samples:
                        layer_samples[module_name] = o.detach().clone().cpu()
                elif isinstance(o, (list, tuple)):
                    for it in o: process_output(it)
            
            process_output(output)
        return hook

    # Register hooks on layers that look like DeltaNet/Recurrent components
    print("  Registering hooks on Gated DeltaNet components...")
    for name, module in model.named_modules():
        # Broader search for Qwen 3.5: linear_attn, delta, ssm, mixer
        if any(x in name.lower() for x in ["delta", "ssm", "recurrent", "linear_attn", "mixer"]):
            module.register_forward_hook(hook_fn(name))

    # 3. Calibration Run
    calib_data = get_calibration_dataset(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN)
    
    print(f"  Running calibration over {NUM_SAMPLES} samples...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(calib_data, total=NUM_SAMPLES)):
            if i >= NUM_SAMPLES: break
            
            # Prepare inputs with batch dimension [1, seq_len]
            input_ids = torch.as_tensor(batch["input_ids"]).unsqueeze(0).to(model.device)
            attention_mask = torch.as_tensor(batch["attention_mask"]).unsqueeze(0).to(model.device)
            
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    # 4. Importance Computation & Position-Aware Pruning
    print("\n📊 Importance Analysis & Pruning Stress Test")
    print(f"  {'Layer Name':<40} {'Base Var':<12} {'Sparsity':<10} {'CosSim'}")
    print("  " + "─" * 75)

    for name, stats in layer_stats.items():
        variance = stats.get_variance()
        if variance is None: continue # Skip if no tensors were captured for this module
        
        importance = variance.abs()
        
        # Determine position-aware sparsity target
        try:
            layer_idx = 0
            parts = name.split('.')
            for p in parts:
                if p.isdigit():
                    layer_idx = int(p)
                    break
        except:
            layer_idx = 0
            
        target_sparsity = 0.40 if layer_idx < 12 else 0.55
        
        # Evaluate pruning on the importance mask
        threshold = torch.quantile(importance.flatten().float(), target_sparsity).item()
        mask = (importance >= threshold).cpu() # Move to CPU
        
        # Validate on sample state (already on CPU)
        sample = layer_samples[name].cpu()
        pruned_sample = sample * mask
        
        cos_sim = torch.nn.functional.cosine_similarity(
            sample.flatten().float().unsqueeze(0),
            pruned_sample.flatten().float().unsqueeze(0)
        ).item()
        
        avg_importance = importance.mean().item()
        verdict = "✅" if cos_sim > 0.98 else "🟡"
        print(f"  {name[:38]:<40} {avg_importance:>12.4f} {target_sparsity*100:>8.0f}%    {cos_sim:>8.4f} {verdict}")

    print("\n✅ Calibration Complete. Use these importance masks in UltimateHybridCache.")

if __name__ == "__main__":
    main()
