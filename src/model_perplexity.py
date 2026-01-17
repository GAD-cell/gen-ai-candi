import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
nfes_to_test = [8, 16, 32, 64, 128, 256] 
num_samples_per_nfe = 20  
generation_length = 512
checkpoint_path = "checkpoints/candi_cold_codeparrot_step_5000.pt"

from src.model.model import CandiLlama, llama_config

model = CandiLlama(llama_config, device=device).to(device)
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
eval_model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
eval_model.eval()

def calculate_perplexity(text, model, tokenizer):
    if not text or len(text.strip()) == 0:
        return None
    
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    if input_ids.size(1) > 1024:
        input_ids = input_ids[:, :1024]
        
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
    return torch.exp(loss).item()

results = {}

for nfe in nfes_to_test:
    print(f"\nÉvaluation NFE = {nfe}")
    ppl_list = []
    
    for _ in tqdm(range(num_samples_per_nfe), desc=f"NFE {nfe}"):
        with torch.no_grad():
            outputs_ids = model.generate(nfe=nfe, generation_length=generation_length)
            
        generated_text = tokenizer.decode(outputs_ids[0], skip_special_tokens=True)
        
        ppl = calculate_perplexity(generated_text, eval_model, tokenizer)
        if ppl is not None and not math.isinf(ppl):
            ppl_list.append(ppl)
            
    if ppl_list:
        avg_ppl = sum(ppl_list) / len(ppl_list)
        results[nfe] = avg_ppl
        print(f"PPL Moyenne pour NFE {nfe}: {avg_ppl:.2f}")


plt.figure(figsize=(10, 6))
plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='-', linewidth=2, color='#619CFF')
plt.xlabel('Number of Function Evaluations (NFE)', fontsize=12)
plt.ylabel('Generative Perplexity', fontsize=12)
plt.title('Performance Frontier: Gen PPL vs NFE', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xscale('log', base=2) 
plt.xticks(nfes_to_test, nfes_to_test)

plt.savefig('performance_frontier.png')

print("\nNFE | Avg Perplexity")
print("-" * 20)
for nfe, ppl in results.items():
    print(f"{nfe:<4} | {ppl:.2f}")