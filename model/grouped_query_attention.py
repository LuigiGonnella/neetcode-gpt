import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads #query heads
        self.num_kv_heads = num_kv_heads #kv heads --> each kv goes into attention of num_heads / kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False) #joint output still of d_model, but when we divide into heads, each will be of head_dim so that head_dim * num_heads = d_model
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False) #same here, but with num_kv_heads, an we should replicate each kv_head in order to have the same number of heads w.r.t. num_heads. The out dim is num_kv_heads * head_dim, so after I reproduce heach head num_heads/kv_heads time, there will be num_heads head od head_dim again
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False) #same replication as k
        self.output_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        print(Q.shape)
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        Q = Q.reshape((B, T, self.num_heads, self.head_dim)).permute(0, 2, 1, 3)
        K = K.reshape((B, T, self.num_kv_heads, self.head_dim)).permute(0, 2, 1, 3)
        V = V.reshape((B, T, self.num_kv_heads, self.head_dim)).permute(0, 2, 1, 3)
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        K = K.repeat_interleave(self.num_heads // self.num_kv_heads, dim = 1)
        V = V.repeat_interleave(self.num_heads // self.num_kv_heads, dim = 1)
        # 4. Compute scaled dot-product attention with causal mask
        lower_triangular = torch.tril(torch.ones((T, T)))
        mask = lower_triangular == 0

        raw_weights = Q @ K.permute(0, 1, 3, 2) / (K.shape[-1] ** 0.5)
        masked_weights = raw_weights.masked_fill(mask, float("-inf"))

        att_weights = torch.softmax(masked_weights, dim = -1)

        hidden = (att_weights @ V).permute(0, 2, 1, 3) #(B, num_heads, T, head_dim) --> (B, T, num_heads, head_dim))
        # 5. Concatenate heads and apply output projection

        hidden = hidden.reshape((B, T, self.num_heads * self.head_dim))
        # 6. Return rounded output (decimals=4)

        output = self.output_proj(hidden)
        return torch.round(output, decimals = 4)
