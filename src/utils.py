def full_attention_causal_mask(self, 
                        attention_mask, 
                        input_tensor, 
                        cache_position=None, 
                        past_key_values=None, 
                        output_attentions=False):
      
    mask_queries = attention_mask[:, None, :, None].bool()  # (batch, 1, seq_len, 1)
    mask_keys = attention_mask[:, None, None, :].bool()     # (batch, 1, 1, seq_len)
    
    full_mask = mask_queries & mask_keys  # (batch, 1, seq_len, seq_len)
    
    return full_mask

class ReformatModelForDiff():
    def __init__(self, model):
        self.model = model
        self._patch_attention_mechanism()
        
    def _patch_attention_mechanism(self):
        """Patch the attention mechanism to use full attention"""

        if hasattr(self.model, 'transformer'):
            base_model = self.model.transformer
        elif hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model
        
        base_model.__class__._update_causal_mask = full_attention_causal_mask
        print(f"Patched {base_model.__class__.__name__} to use full attention")

    def get_model(self):
        return self.model