import transformers.models.llama.modeling_llama as llama_module
import transformers.modeling_attn_mask_utils as mask_utils
import torch

def bidirectional_mask_replacement(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values=None,
    position_ids=None,
    **kwargs
):
    batch_size, seq_length = input_embeds.shape[0], input_embeds.shape[1]
    dtype = input_embeds.dtype
    device = input_embeds.device
    min_dtype = torch.finfo(dtype).min
    
    if attention_mask is not None:
        mask_4d = attention_mask[:, None, None, :]
        causal_mask = torch.where(mask_4d == 1, 0.0, min_dtype)
    else:
        causal_mask = torch.zeros((batch_size, 1, seq_length, seq_length), dtype=dtype, device=device)
    
    return causal_mask

class ReformatModelForDiff():
    def __init__(self, model):
        self.model = model
        self._patch_attention_mechanism()
        
    def _patch_attention_mechanism(self):
        mask_utils.create_causal_mask = bidirectional_mask_replacement
        llama_module.create_causal_mask = bidirectional_mask_replacement
        
        if hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model
            
        for layer in base_model.layers:
            if hasattr(layer.self_attn, 'is_causal'):
                layer.self_attn.is_causal = False

        print(f"LlamaModel patched for full attention")
    def get_model(self):
        return self.model