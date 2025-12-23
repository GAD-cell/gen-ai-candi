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
    def __init__(self, llama_config, device):
        super().__init__()
        self.llama = LlamaForCausalLM(llama_config)
        self.llama = ReformatModelForDiff(self.llama).get_model()
        self.vocab_size = llama_config.vocab_size

        self.corruption_bias = nn.Parameter(torch.zeros(llama_config.hidden_size))

        self.r_max = 0.499
        self.r_min = 1e-5
        self.corruption_lambd = 0.5

        self.device = device

    def _get_sigma(self,t):
        r_t = (self.r_max - self.r_min) * t + self.r_min
        r_t_np = r_t.cpu().numpy()
        inv_phi = torch.from_numpy(norm.ppf(r_t_np)).to(self.device).float()
        sigma_t = -1 / (inv_phi * torch.sqrt(torch.tensor(2.0)))
        
        return sigma_t
    
    def _get_noising_params(self, t, bsz, seq_len, attention_mask=None):

        if len(t.shape)==1:
            t = t.repeat(bsz,1)
        
        alpha_t = 1 - t
        
        sigma_t = self._get_sigma(t)


        m_t = torch.bernoulli(alpha_t.expand(bsz, seq_len))
        if attention_mask is not None :
            m_t = torch.where(attention_mask == 0, torch.ones_like(m_t), m_t)
  
        return alpha_t, sigma_t, m_t


    def _get_noising_kernel(self, input_ids, attention_mask):
        bsz, seq_len = input_ids.shape
        device = input_ids.device


        t = torch.rand(bsz, 1, device=device)
        
        alpha_t, sigma_t, m_t = self._get_noising_params(t, bsz, seq_len, attention_mask)
        
        x_0_one_hot = F.one_hot(input_ids, num_classes=self.vocab_size).float() 
        noise = torch.randn_like(x_0_one_hot)
        sigma_mask = attention_mask.unsqueeze(-1).float()
        x_tilde = x_0_one_hot + (sigma_t.unsqueeze(-1) * sigma_mask) * noise

        return alpha_t, m_t, x_tilde
    
    def forward(self, input_ids, attention_mask, m_t=None, labels=None):

        if self.training:
            alpha_t, m_t, x_tilde_one_hot = self._get_noising_kernel(input_ids, attention_mask)
            
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
                outputs.loss = (-target_log_probs/ (1.0 - alpha_t)).mean()
            return outputs  
                     
        else : 
            outputs = self.llama(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
                )
            return outputs
    
    def generate(self, inputs_ids=None, attention_mask=None, nfe=8, generation_length=128, batch_size=1):
        
        t = torch.Tensor([1.0])
        alpha_t, sigma_t, m_t = self._get_noising_params(t, batch_size, generation_length,)
        noise = torch.randn((batch_size,generation_length,self.vocab_size))
        x_tilde_one_hot = sigma_t.unsqueeze(-1) * noise
        # Initial predictions
        x_t = torch.argmax(x_tilde_one_hot,dim=-1)

        W_embed = self.llama.get_input_embeddings().weight
        # Initial noisy embeddings
        y_cache = (1 - self.corruption_lambd ) * torch.matmul(x_tilde_one_hot, W_embed) + self.corruption_lambd  * self.corruption_bias 

        t_steps = torch.arange(1,-1/nfe,-1/nfe)
        for t in t_steps :
            y_t = torch.where(m_t.unsqueeze(-1)==1, self.llama.get_input_embeddings()(x_t), y_cache)
            # x_s_new from learned posterior 
            x_s_new = self.llama(
                inputs_embeds=y_t,
                attention_mask=attention_mask,
                return_dict=True
            ).logits

            # mask randomly to create anchor points
            u = torch.rand((batch_size,generation_length))
            alpha_s = alpha_t - 1/nfe
            m_s_new = u < (alpha_s - alpha_t) / (1 - alpha_t)
            
            # score function
            x_hat_s = torch.argmax(x_s_new,dim=-1)
            grad_logp = - (y_t - self.llama.get_input_embeddings()(x_hat_s)) / (sigma_t)**2

            # ODE update
            s = (t-1/nfe).repeat(batch_size,1)
            sigma_s = self._get_sigma(s)
            y_cache = y_t - (sigma_t - sigma_s) * grad_logp

            m_s_prime = (m_s_new) & (m_t == 0)
            x_t = torch.where(m_t == 1, x_t, torch.where(m_s_prime, x_hat_s, x_t))
            m_t = torch.logical_or(m_t, m_s_new).float()


        return x_t
    

