from src.model.model import CandiLlama, llama_config
import torch
from transformers import GPT2Tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
model = CandiLlama(llama_config,  device=device).to(device)
model.train()

dummy_input = torch.randint(0, 50257, (2, 128)).to(device)
dummy_mask = torch.ones_like(dummy_input)

outputs = model(dummy_input, attention_mask=dummy_mask, labels=dummy_input)
print(f"Test Forward. output shape : {outputs.logits.shape}")

# generation 
outputs_ids = model.generate()
print(f"Test Generation. output : {outputs}")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = tokenizer.batch_decode(outputs_ids,skip_special_tokens=True)
print(text)