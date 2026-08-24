import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.wK = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.wQ = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.wV = nn.Linear(embedding_dim, attention_dim, bias = False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        

        K = self.wK(embedded) #(B, N, attention_dim)
        Q = self.wQ(embedded) #(B, N, attention_dim)
        V = self.wV(embedded) #(B, N, attention_dim)

        N = K.shape[1]
        attention_dim = K.shape[2]

        Kt = torch.transpose(K, 1, 2)
        mask = torch.tril(torch.ones((N,N))) == 0 #gives True in upper tri, which must be masked out
        att_logits = ((Q @ Kt) / (attention_dim ** (1/2))).masked_fill(mask, float("-inf")) #(B, N, N)
        att_weights = nn.functional.softmax(att_logits, dim = 2) #probs across last dim --> each token (row) pays full attention (to all tokens, cols). When -inf result is 0.
        #(B, N, N)
        out = att_weights @ V #(B, N, attention_dim)

        return torch.round(out, decimals = 4)
