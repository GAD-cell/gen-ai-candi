import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from src.utils import ReformatModelForDiff
from torch.distributions.normal import Normal

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

def sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)

def gaussian_to_disc_corruption(K: int, sigma: float) -> float:
    normal = Normal(0., 1.)
    s = torch.linspace(-4*sigma, 1+6*sigma, 1001)
    phi = normal.log_prob((s-1)/sigma).exp() / sigma
    cdf_power = normal.cdf(s/sigma).pow(K-1)
    return 1.0 - torch.trapz(cdf_power * phi, s).item()

def build_error_to_sigma_schedule(vocab_size, sigma_range=(0.01, 2.0), num_points=500, device='cpu'):
    sigmas = torch.linspace(*sigma_range, steps=num_points, device=device)
    errors = torch.tensor(
        [gaussian_to_disc_corruption(vocab_size, s.item()) for s in sigmas], 
        device=device
    )
    return sigmas, errors

def sigma_from_time_vectorized(t, sigmas, errors):
    t = t.clamp(min=0.0, max=1.0)
    indices = torch.searchsorted(errors, t, right=True).clamp(1, len(errors) - 1)
    i0 = indices - 1
    i1 = indices

    e0 = errors[i0]
    e1 = errors[i1]
    s0 = sigmas[i0]
    s1 = sigmas[i1]

    interp_t = (t - e0) / (e1 - e0 + 1e-8)
    return s0 + interp_t * (s1 - s0)

class CandiLlama(nn.Module):
    def __init__(self, llama_config, device):
        super().__init__()
        self.device = device
        
        self.llama = LlamaForCausalLM(llama_config)
        self.llama = ReformatModelForDiff(self.llama).get_model()
        
        original_vocab_size = llama_config.vocab_size
        self.vocab_size = original_vocab_size + 1 
        self.mask_index = original_vocab_size
        
        self.llama.resize_token_embeddings(self.vocab_size)
        
        self.corruption_bias = nn.Parameter(torch.zeros(llama_config.hidden_size))
        self.corruption_lambd = 0.5
        
        self.temp = 1.0
        self.sigma_min = 0.01
        self.sigma_max = 2.0 
        self.min_percentile = 0.01
        self.max_percentile = 0.45 
        self.is_embed = False
        
        sigmas, errors = build_error_to_sigma_schedule(
            original_vocab_size, 
            sigma_range=(self.sigma_min, self.sigma_max), 
            device=device
        )
        self.register_buffer('sigmas_table', sigmas)
        self.register_buffer('errors_table', errors)

        self.to(device)

    def get_continuous_from_discrete_noise(self, t_discrete):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** (t_discrete)

    def get_inference_sigma(self, t_linear):
        target_percentile = t_linear * (self.max_percentile - self.min_percentile) + self.min_percentile
        return sigma_from_time_vectorized(target_percentile, self.sigmas_table, self.errors_table)
    
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
            logits_for_loss = outputs.logits[:, :, :self.mask_index]
            
            log_probs = F.log_softmax(logits_for_loss, dim=-1)
            target_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            
            weight = torch.clamp(1.0 / (1 - alpha_t + 1e-8), max=10.0)
            outputs.loss = (-target_log_probs).mean()
            
        return outputs

    @torch.no_grad()
    def _continuous_step(
        self,
        x: torch.Tensor,
        time_t: torch.Tensor,
        time_s: torch.Tensor,
        sigma_s: torch.Tensor,
        sigma_t: torch.Tensor,
        clean_mask: torch.Tensor = None,
        is_embed=False,
    ) -> torch.Tensor:
        
        sigma_t_vec = torch.ones(x.shape[0], device=x.device) * sigma_t.item()
        dt_cont_vec = torch.ones(x.shape[0], device=x.device) * (sigma_s - sigma_t).item()
        
        if clean_mask is None:
            clean_mask = torch.zeros(x.shape[:-1], device=x.device)

        W_embed = self.llama.get_input_embeddings().weight
        
        if self.is_embed:
            inputs_embeds = x
        else:
            inputs_embeds = torch.matmul(x, W_embed)

        sigma_reshaped = sigma_t_vec.view(-1, 1, 1)
        inputs_embeds = self.precondition_inputs(inputs_embeds, clean_mask, sigma_reshaped)

        outputs = self.llama(
            inputs_embeds=inputs_embeds, 
            return_dict=True
        )
        
        logits = outputs.logits / self.temp
        
        logits_voc = logits[:, :, :self.mask_index]
        p_x0_voc = torch.softmax(logits_voc, dim=-1)
        
        p_x0_full = torch.zeros((x.shape[0], x.shape[1], self.vocab_size), device=x.device)
        p_x0_full[:, :, :self.mask_index] = p_x0_voc

        if self.is_embed:
            x0_hat = torch.matmul(p_x0_full, W_embed)
        else:
            x0_hat = p_x0_full

        d = (x - x0_hat) / (sigma_t_vec[:, None, None] ** 2 + 1e-8)
        x_cont = x - dt_cont_vec[:, None, None] * d
        
        return x_cont, p_x0_voc

    def _discrete_step(self, x_sigma, p_x0, t, dt, prev_clean_mask, noise_removal_step=False):
        if noise_removal_step:
            s = 0.0
        else:
            s = t - dt

        t_vec = torch.ones(x_sigma.shape[0], device=x_sigma.device) * t.item()
        s_vec = torch.ones(x_sigma.shape[0], device=x_sigma.device) * s.item()
        
        mask_probs = torch.ones((x_sigma.shape[0], x_sigma.shape[1], 1), device=x_sigma.device) * s
        
        unmasked_probs = p_x0 * (t_vec - s_vec)[:, None, None]

        q_xs = torch.cat([unmasked_probs, mask_probs], dim=-1)
        
        _x = sample_categorical(q_xs)
        
        mask_idx_in_q = self.mask_index
        
        new_clean_mask = (prev_clean_mask.bool() | (_x != mask_idx_in_q)).float()

        old_x_tokens = x_sigma.argmax(dim=-1)

        sampled_real_tokens = torch.where(_x != mask_idx_in_q, _x, old_x_tokens)
        
        updated_tokens = torch.where(prev_clean_mask.bool(), old_x_tokens, sampled_real_tokens)
        
        #num_classes doit être self.vocab_size
        updated_x = torch.nn.functional.one_hot(updated_tokens, num_classes=self.vocab_size).float().to(x_sigma.device)

        updated_x = updated_x * new_clean_mask.unsqueeze(-1) + (1 - new_clean_mask).unsqueeze(-1) * x_sigma

        return updated_x, new_clean_mask


    @torch.no_grad()
    def generate(self, prompt_ids=None, nfe=64, generation_length=128, batch_size=1):
        device = self.device
        num_tokens = generation_length
        
        x = torch.randn((batch_size, num_tokens, self.vocab_size), device=device)
        
        if self.is_embed:
            x = torch.matmul(x, self.llama.get_input_embeddings().weight)
            
        clean_mask = torch.zeros((batch_size, num_tokens), device=device)

        if prompt_ids is not None:
            prompt_len = prompt_ids.shape[1]
            
            prompt_mask = torch.zeros((batch_size, num_tokens), device=device)
            prompt_mask[:, :prompt_len] = 1.0
            
            clean_mask = torch.max(clean_mask, prompt_mask)
            
            if self.is_embed:
                prompt_embeds = self.llama.get_input_embeddings()(prompt_ids)
                padding_embeds = torch.zeros((batch_size, num_tokens - prompt_len, self.config.hidden_size), device=device)
                full_prompt_embeds = torch.cat([prompt_embeds, padding_embeds], dim=1)
                x = prompt_mask.unsqueeze(-1) * full_prompt_embeds + (1 - prompt_mask.unsqueeze(-1)) * x
            else:
                prompt_one_hot = torch.nn.functional.one_hot(prompt_ids, num_classes=self.vocab_size).float()
                padding_one_hot = torch.zeros((batch_size, num_tokens - prompt_len, self.vocab_size), device=device)
                full_prompt_one_hot = torch.cat([prompt_one_hot, padding_one_hot], dim=1)
                x = prompt_mask.unsqueeze(-1) * full_prompt_one_hot + (1 - prompt_mask.unsqueeze(-1)) * x

        timesteps = torch.linspace(0.999, 1e-5, nfe + 1, device=device)
        continuous_noise = self.get_continuous_from_discrete_noise(timesteps)
        dt = (1 - 1e-5) / nfe

        for i in range(nfe):
            t = timesteps[i]
            s = timesteps[i+1]
            
            sigma_s = continuous_noise[i]
            sigma_t = continuous_noise[i+1]

            x_cont, p_x0_voc = self._continuous_step(
                x, t, 
                sigma_s=sigma_s, 
                sigma_t=sigma_t, 
                clean_mask=clean_mask, 
                time_s=s
            )
            
            x, clean_mask = self._discrete_step(
                x_cont, 
                p_x0_voc, 
                t, 
                dt, 
                prev_clean_mask=clean_mask
            )

        final_tokens = x.argmax(dim=-1)
        return final_tokens