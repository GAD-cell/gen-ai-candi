import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.stats import norm
from transformers import LlamaConfig, LlamaForCausalLM
from src.utils import ReformatModelForDiff

llama_config = LlamaConfig(
    vocab_size=50257,
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=10,
    num_attention_heads=12,
    max_position_embeddings=1024,
    initializer_range=0.02,

    tie_word_embeddings=False 
)

class CandiLlama(nn.Module):
    def __init__(self, llama_config):
        super().__init__()
        self.llama = LlamaForCausalLM(llama_config)
        self.llama = ReformatModelForDiff(self.llama).get_model()
        self.vocab_size = llama_config.vocab_size

        self.corruption_bias = nn.Parameter(torch.zeros(llama_config.hidden_size))

        self.r_max = 0.499
        self.r_min = 1e-5
        self.corruption_lambd = 0.5
    
    def _get_kernel_noising(self, input_ids, attention_mask):
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        t = torch.rand(bsz, 1, device=device)
        alpha_t = 1 - t
        
        r_t = (self.r_max - self.r_min) * t + self.r_min
        r_t_np = r_t.cpu().numpy()
        inv_phi = torch.from_numpy(norm.ppf(r_t_np)).to(device).float()
        sigma_t = -1 / (inv_phi * torch.sqrt(torch.tensor(2.0)))

        m_t = torch.bernoulli(alpha_t.expand(bsz, seq_len))
        m_t = torch.where(attention_mask == 0, torch.ones_like(m_t), m_t)

        x_0_one_hot = F.one_hot(input_ids, num_classes=self.vocab_size).float() 
        noise = torch.randn_like(x_0_one_hot)
        sigma_mask = attention_mask.unsqueeze(-1).float()
        x_tilde = x_0_one_hot + (sigma_t.unsqueeze(-1) * sigma_mask) * noise

        return alpha_t, m_t, x_tilde
    
    def forward(self, input_ids, attention_mask, m_t=None, labels=None):

        if self.training:
            alpha_t, m_t, x_tilde_one_hot = self._get_kernel_noising(input_ids, attention_mask)
            
            W_embed = self.llama.get_input_embeddings().weight
            x_tilde_embed = torch.matmul(x_tilde_one_hot, W_embed)
            x_corrupted = (1 - self.corruption_lambd ) * x_tilde_embed + self.corruption_lambd  * self.corruption_bias
            x_0_embed = self.llama.get_input_embeddings()(input_ids)
            inputs_embeds = torch.where(m_t.unsqueeze(-1) == 1, x_0_embed, x_corrupted)
            
            outputs = self.llama(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            if labels is not None :
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                target_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                print(target_log_probs.shape)
                outputs.loss = (-target_log_probs/ (1.0 - alpha_t)).mean()
            return outputs  
                     
        else : 
            outputs = self.llama(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
                )
            return outputs
    

