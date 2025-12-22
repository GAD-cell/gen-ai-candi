from src.model.model import CandiLlama, llama_config
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CandiLlama(llama_config).to(device)
model.train()

dummy_input = torch.randint(0, 50257, (2, 128)).to(device)
dummy_mask = torch.ones_like(dummy_input)

logits = model(dummy_input, attention_mask=dummy_mask)
print(f"Test Forward réussi. Shape des logits : {logits.shape}")