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
    num_hidden_layers=12,
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
    
    def precondition_inputs(self, y_t, m_t, t):
        sigma_t = self._get_sigma(t).view(-1, 1, 1)
        
        inv_scale = 1.0 / torch.sqrt(sigma_t**2 + 1.0)
        
        m_t_ext = m_t.unsqueeze(-1)
        
        y_t_scaled = torch.where(
            m_t_ext == 1,
            y_t,               
            y_t * inv_scale    
        )
        
        return y_t_scaled

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

        return alpha_t, m_t, x_tilde, t
    
    def forward(self, input_ids, attention_mask, m_t=None, labels=None):

        if self.training:
            alpha_t, m_t, x_tilde_one_hot, t = self._get_noising_kernel(input_ids, attention_mask)
            
            W_embed = self.llama.get_input_embeddings().weight
            x_tilde_embed = torch.matmul(x_tilde_one_hot, W_embed)
            x_corrupted = (1 - self.corruption_lambd ) * x_tilde_embed + self.corruption_lambd  * self.corruption_bias
            x_0_embed = self.llama.get_input_embeddings()(input_ids)
            inputs_embeds = torch.where(m_t.unsqueeze(-1) == 1, x_0_embed, x_corrupted)
            inputs_embeds = self.precondition_inputs(inputs_embeds, m_t, t)

            outputs = self.llama(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )

            if labels is not None :
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                target_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                weight = torch.clamp(1.0 / (1-alpha_t + 1e-8), max=100.0)
                outputs.loss = (-target_log_probs * weight).mean()
            return outputs  
                     
        else:
            alpha_t, m_t, x_tilde_one_hot, t = self._get_noising_kernel(input_ids, attention_mask)
            
            W_embed = self.llama.get_input_embeddings().weight
            x_tilde_embed = torch.matmul(x_tilde_one_hot, W_embed)
            x_corrupted = (1 - self.corruption_lambd) * x_tilde_embed + self.corruption_lambd * self.corruption_bias
            x_0_embed = self.llama.get_input_embeddings()(input_ids)
            
            eval_inputs_embeds = torch.where(m_t.unsqueeze(-1) == 1, x_0_embed, x_corrupted)
            eval_inputs_embeds = self.precondition_inputs(eval_inputs_embeds, m_t, t)
            
            outputs = self.llama(
                inputs_embeds=eval_inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            if labels is not None:
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                target_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                weight = torch.clamp(1.0 / (1-alpha_t + 1e-8), max=100.0)
                outputs.loss = (-target_log_probs * weight).mean()
                
            return outputs
    
    def generate(self, prompt_ids=None, nfe=8, generation_length=128, batch_size=1):
        device = self.corruption_bias.device
        
        if prompt_ids is not None:
            prompt_len = prompt_ids.shape[1]
            x_t = torch.cat([
                prompt_ids, 
                torch.zeros((batch_size, generation_length - prompt_len), dtype=torch.long, device=device)
            ], dim=1)
            is_prompt = torch.zeros((batch_size, generation_length), device=device)
            is_prompt[:, :prompt_len] = 1
        else:
            x_t = torch.zeros((batch_size, generation_length), dtype=torch.long, device=device)
            is_prompt = torch.zeros((batch_size, generation_length), device=device)

        t_start = torch.ones((batch_size, 1), device=device)
        alpha_t, sigma_t, m_t = self._get_noising_params(t_start, batch_size, generation_length)
        
        m_t = torch.logical_or(m_t, is_prompt).float()
        
        noise = torch.randn((batch_size, generation_length, self.vocab_size), device=device)
        x_tilde_one_hot = sigma_t.view(batch_size, 1, 1) * noise
        
        W_embed = self.llama.get_input_embeddings().weight
        x_tilde_embed = torch.matmul(x_tilde_one_hot, W_embed)
        y_cache = (1 - self.corruption_lambd) * x_tilde_embed + self.corruption_lambd * self.corruption_bias 

        t_steps = torch.linspace(1, 0, nfe + 1).to(device)
        
        for i in range(len(t_steps) - 1):
            t_curr = t_steps[i]
            t_next = t_steps[i+1]
            
            y_t = torch.where(m_t.unsqueeze(-1) == 1, self.llama.get_input_embeddings()(x_t), y_cache)
            y_t_input = self.precondition_inputs(y_t, m_t, t_curr.repeat(batch_size, 1))

            outputs = self.llama(
                inputs_embeds=y_t_input,
                return_dict=True
            )
            x_s_logits = outputs.logits
            x_hat_s = torch.argmax(x_s_logits, dim=-1)
            
            if prompt_ids is not None:
                x_hat_s = torch.where(is_prompt == 1, x_t, x_hat_s)

            sigma_t_val = self._get_sigma(torch.full((batch_size, 1), t_curr, device=device)).view(-1, 1, 1)
            sigma_s_val = self._get_sigma(torch.full((batch_size, 1), t_next, device=device)).view(-1, 1, 1)
            
            grad_logp = - (y_t - self.llama.get_input_embeddings()(x_hat_s)) / (sigma_t_val**2 + 1e-8)
            
            diff_variance = 0.5 * (sigma_t_val**2 - sigma_s_val**2)
            y_cache = y_t - diff_variance * grad_logp

            alpha_curr = 1 - t_curr
            alpha_next = 1 - t_next
            
            prob_transition = (alpha_next - alpha_curr) / (1 - alpha_curr + 1e-8)
            u = torch.rand((batch_size, generation_length), device=device)
            
            m_s_new = (u < prob_transition) | (is_prompt == 1)
            
            m_s_prime = m_s_new & (m_t == 0)

            x_t = torch.where(m_t == 1, x_t, torch.where(m_s_prime, x_hat_s, x_t))
            m_t = torch.logical_or(m_t, m_s_new).float()

        return x_t

