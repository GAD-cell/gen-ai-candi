from src.model.model import CandiLlama, llama_config
import torch
from transformers import GPT2Tokenizer
import os 

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CandiLlama(llama_config,  device=device).to(device)
checkpoint_path = "checkpoints/candi_code_step_1000.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

model.train()

dummy_input = torch.randint(0, 50257, (2, 128)).to(device)
dummy_mask = torch.ones_like(dummy_input)

outputs = model(dummy_input, attention_mask=dummy_mask, labels=dummy_input)
print(f"Test Forward. output shape : {outputs.logits.shape}")


# generation
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()
prompt_text="def add(a,b):"
prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
outputs_ids = model.generate(prompt_ids=prompt_ids,generation_length=512)
print(f"Test Generation. output : {outputs}")
text = tokenizer.batch_decode(outputs_ids,skip_special_tokens=True)
print(text)