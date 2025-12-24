import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
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
        self.corruption_lambd = 0.5
        
        self.temp = 0.1 
        self.sigma_min = 0.01
        self.sigma_max = 2.0 
        self.min_percentile = 0.01
        self.max_percentile = 0.25

        self.device = device
        self.to(device)

    def get_continuous_from_discrete_noise(self, t_discrete):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t_discrete

    def get_inference_sigma(self, t_linear):
        target_percentile = t_linear * (self.max_percentile - self.min_percentile) + self.min_percentile
        return self.get_continuous_from_discrete_noise(target_percentile)
    
    def precondition_inputs(self, y_t, m_t, sigma_t):
        inv_scale = 1.0 / torch.sqrt(sigma_t**2 + 1.0)
        
        y_t_scaled = torch.where(
            m_t.unsqueeze(-1) == 1,
            y_t,               
            y_t * inv_scale    
        )
        return y_t_scaled

    def _get_noising_kernel(self, input_ids, attention_mask):
        bsz, seq_len = input_ids.shape
        t = torch.rand(bsz, 1, device=input_ids.device)
        
        alpha_t = 1 - t
        m_t = torch.bernoulli(alpha_t.expand(bsz, seq_len))
        if attention_mask is not None:
            m_t = torch.where(attention_mask == 0, torch.ones_like(m_t), m_t)
        sigma_t = self.get_continuous_from_discrete_noise(t)
        
        x_0_one_hot = F.one_hot(input_ids, num_classes=self.vocab_size).float() 
        noise = torch.randn_like(x_0_one_hot)        
        x_tilde = x_0_one_hot + sigma_t.view(bsz, 1, 1) * noise

        return alpha_t, m_t, x_tilde, t

    def forward(self, input_ids, attention_mask, labels=None):
        alpha_t, m_t, x_tilde_one_hot, t = self._get_noising_kernel(input_ids, attention_mask)
        
        W_embed = self.llama.get_input_embeddings().weight
        x_tilde_embed = torch.matmul(x_tilde_one_hot, W_embed)
        
        x_corrupted = (1 - self.corruption_lambd) * x_tilde_embed + self.corruption_lambd * self.corruption_bias
        x_0_embed = self.llama.get_input_embeddings()(input_ids)
        
        inputs_embeds = torch.where(m_t.unsqueeze(-1) == 1, x_0_embed, x_corrupted)
        
        sigma_t = self.get_continuous_from_discrete_noise(t).view(-1, 1, 1)
        inputs_embeds = self.precondition_inputs(inputs_embeds, m_t, sigma_t)

        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )

        outputs.logits = outputs.logits / self.temp

        if labels is not None:
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            target_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            
            weight = torch.clamp(1.0 / (1 - alpha_t + 1e-8), max=10.0)
            outputs.loss = (-target_log_probs * weight).mean()
            
        return outputs

    @torch.no_grad()
    def generate(self, prompt_ids=None, nfe=64, generation_length=128, batch_size=1):
        device = self.device
        
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
        sigma_start = self.get_inference_sigma(t_start).view(batch_size, 1, 1)
        
        noise = torch.randn((batch_size, generation_length, self.vocab_size), device=device)
        x_tilde_embed = torch.matmul(noise, self.llama.get_input_embeddings().weight)
        
        y_cache = (1 - self.corruption_lambd) * (sigma_start * x_tilde_embed) + self.corruption_lambd * self.corruption_bias
        
        m_t = is_prompt.clone().float()

        t_steps = torch.linspace(1, 0, nfe + 1, device=device)
        
        for i in range(len(t_steps) - 1):
            t_curr = t_steps[i]
            t_next = t_steps[i+1]
            
            sigma_t = self.get_inference_sigma(t_curr).view(-1, 1, 1)
            sigma_s = self.get_inference_sigma(t_next).view(-1, 1, 1)
            
            y_t = torch.where(m_t.unsqueeze(-1) == 1, self.llama.get_input_embeddings()(x_t), y_cache)
            y_t_input = self.precondition_inputs(y_t, m_t, sigma_t)

            outputs = self.llama(inputs_embeds=y_t_input, return_dict=True)
            logits = outputs.logits / self.temp
            x_hat_s = torch.argmax(logits, dim=-1)
            
            if prompt_ids is not None:
                x_hat_s = torch.where(is_prompt == 1, x_t, x_hat_s)

            x0_hat_embed = self.llama.get_input_embeddings()(x_hat_s)
            score = (y_t - x0_hat_embed) / (sigma_t**2 + 1e-8)
            dt_sigma = sigma_t - sigma_s
            y_cache = y_t - dt_sigma * score


            prob_transition = (t_curr - t_next) / (t_curr + 1e-8)
            
            u = torch.rand((batch_size, generation_length), device=device)
            m_s_new = (u < prob_transition) | (is_prompt == 1)
            m_s_prime = m_s_new & (m_t == 0)

            x_t = torch.where(m_t == 1, x_t, torch.where(m_s_prime, x_hat_s, x_t))
            m_t = torch.logical_or(m_t, m_s_new).float()

        return x_t