import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    #!on higher level I should call the attention with X being the whole BATCH SEQUENCES if prefill
    #!and x being JUST the LAST BATCH TOKEN in decode, so that after computing K and V they are actually (B, 1, d_model) and not (B, C, d_model)
    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        if self.cache_k is None:
            self.cache_k = new_k
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim = 1) #new token in seq_len
        
        if self.cache_v is None:
            self.cache_v = new_v
        else:
            self.cache_v = torch.cat([self.cache_v, new_v], dim = 1) #new token in seq_len
        
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        Q = self.q_proj(x) #(B, C, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # 2. If kv_cache is None, create a new KVCache
        if kv_cache is None:
            kv_cache = KVCache()
        
        # 3. Update the cache with the new K and V
        cache_k, cache_v = kv_cache.update(K, V)

        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        raw_weights = (Q @ cache_k.permute(0, 2, 1)) / (K.shape[2] ** 0.5)
        # 5. Apply a causal mask offset by the number of previously cached tokens
        new_len = Q.shape[1]
        total_len = cache_k.shape[1]
        past_len = total_len - new_len #new query must look at past_len, past_len + 1, ..., tokens

        #raw_weights: (B, new_len, total_len) --> (B, C, C) in prefill and (B, 1, C+1) in decode

        query_positions = torch.arange(
            past_len,
            total_len,
            device=x.device
        )  # (new_len,) --> from past_len to total_len

        key_positions = torch.arange(
            total_len,
            device=x.device
        )  # (total_len,) 

        mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        #(1, total_len) > (new_len, 1) gives (new_len, total_len) with true on all cells (key, query) where key > query --> that query cannot look at that future key

        #--> if decode with only 1 token --> no mask at all
        #--> if decode with 2 tokens --> last key token masked to the first (of these two) query token
        #--> if decode with 3 tokens --> last 2 keys maske to first query, last key masked to before-last query
        #...

        raw_weights = raw_weights.masked_fill(mask, float("-inf")) #-inf in future tokens so when computing softmax they go to 0 in exp

        att_weights = torch.softmax(raw_weights, dim = -1) #probs on a row, over columns

        output = att_weights @ cache_v
        # 6. Return (rounded output, kv_cache)
        return torch.round(output, decimals=4), kv_cache
