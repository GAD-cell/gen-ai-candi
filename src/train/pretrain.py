import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig, get_cosine_schedule_with_warmup
from src.model.model import CandiLlama, llama_config
import torch.nn as nn
import os
from tqdm import tqdm
import wandb

total_steps = 10000
warmup_steps = int(0.1 * total_steps)
learning_rate = 1e-4
weight_decay = 1e-3
batch_size = 16
generation_interval = 50  


class CodeParrotDataset(IterableDataset):
    def __init__(self, tokenizer, context_length=1024, cache_dir=None):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.cache_dir = cache_dir
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Chargement du dataset CodeParrot en streaming
        self.dataset = load_dataset(
            "codeparrot/codeparrot-clean-train",
            split="train",
            streaming=True,
            cache_dir=self.cache_dir
        )

    def __iter__(self):
        for example in self.dataset:
            # On utilise le champ 'content' qui contient le code
            outputs = self.tokenizer(
                example["content"],
                truncation=True,
                max_length=self.context_length,
                padding="max_length",
                return_tensors="pt"
            )
            yield {
                "input_ids": outputs["input_ids"].squeeze(0),
                "attention_mask": outputs["attention_mask"].squeeze(0)
            }

class Text8Dataset(IterableDataset):
    def __init__(self, tokenizer, context_length=1024, cache_dir=None):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.cache_dir = cache_dir
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.dataset = load_dataset(
            "afmck/text8-chunked1024",
            split="train",
            streaming=True,
            cache_dir=self.cache_dir
        )

    def __iter__(self):
        for example in self.dataset:
            outputs = self.tokenizer(
                example["text"],
                truncation=True,
                max_length=self.context_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            yield {
                "input_ids": outputs["input_ids"].squeeze(0),
                "attention_mask": outputs["attention_mask"].squeeze(0)
            }


CACHE_DIR = "./data/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

context_length = 1024
train_dataset = CodeParrotDataset(
    tokenizer, 
    context_length=context_length,
    cache_dir=CACHE_DIR
)
train_loader = DataLoader(train_dataset, batch_size=batch_size)


def train_step(model, optimizer, scheduler, batch, device):
    optimizer.zero_grad()
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    outputs = model(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        labels=input_ids
    )
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    return loss.item()


def generate_and_log(model, tokenizer, step, device, num_samples=2, generation_length=128, nfe=8):
    """Génère des échantillons et les log dans WandB"""
    model.eval()
    
    with torch.no_grad():
        generated_ids = model.generate(nfe=nfe)
        
        generated_ids = generated_ids.cpu()
        
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        tqdm.write(f"\n{'='*80}")
        tqdm.write(f"Generated Samples at Step {step}:")
        tqdm.write(f"{'='*80}")
        for idx, text in enumerate(generated_texts):
            tqdm.write(f"\n--- Sample {idx+1} ---")
            tqdm.write(text[:500])  
            tqdm.write("")
        tqdm.write(f"{'='*80}\n")
    
    model.train()


device = "cuda" if torch.cuda.is_available() else "cpu"


wandb.init(
    project="candi-llama-pretraining",
    name="pretrain-codeparrot",
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "context_length": context_length,
        "total_steps": total_steps,
    }
)

model = CandiLlama(llama_config, device=device).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

wandb.watch(model, log="all", log_freq=100)

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

model.train()

running_loss = 0.0
log_interval = 10

for epoch in range(1):
    pbar = tqdm(enumerate(train_loader), total=total_steps, desc=f"Epoch {epoch+1}")
    
    for i, batch in pbar:
        if i >= total_steps:
            break
            
        loss = train_step(model, optimizer, scheduler, batch, device)
        running_loss += loss
        
        current_lr = scheduler.get_last_lr()[0]
        
        wandb.log({
            "loss": loss,
            "learning_rate": current_lr,
            "step": i,
            "epoch": epoch
        })
        
        if i % log_interval == 0 and i > 0:
            avg_loss = running_loss / log_interval
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'step': i
            })
            
            wandb.log({
                "avg_loss": avg_loss,
                "step": i
            })
            
            running_loss = 0.0
        
        if i % generation_interval == 0 and i > 0:
            generate_and_log(
                model=model,
                tokenizer=tokenizer,
                step=i,
                device=device,
                num_samples=1,
                generation_length=128,
                nfe=16
            )
            
        if i % 1000 == 0 and i > 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"candi_codeparrot_step_{i}.pt")
            
            torch.save({
                'step': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            
            tqdm.write(f"✓ Checkpoint saved: {checkpoint_path}")
            
            wandb.save(checkpoint_path)
    
    pbar.close()


final_checkpoint_path = os.path.join(CHECKPOINT_DIR, "candi_codeparrot_final.pt")
torch.save({
    'step': total_steps,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
}, final_checkpoint_path)
tqdm.write(f"✓ Final checkpoint saved: {final_checkpoint_path}")
wandb.save(final_checkpoint_path)

wandb.finish()