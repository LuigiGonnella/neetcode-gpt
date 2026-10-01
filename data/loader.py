import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        #
        # Return (X, Y) where:
        # - X has shape (batch_size, context_length)
        # - Y has shape (batch_size, context_length)
        # - Y is X shifted right by 1 (Y[i][j] = data[start_i + j + 1])
        #
        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions
        
        torch.manual_seed(0)
        n = data.shape[0]

        # X = torch.empty((batch_size, context_length), dtype=data.dtype, device=data.device)
        # Y = torch.empty((batch_size, context_length), dtype=data.dtype, device=data.device)

        starts = torch.randint(low=0, high=n - context_length,size=(batch_size,)
) #(batch_size, ) random ints

        X = torch.stack([data[start: start+context_length] for start in starts])
        Y = torch.stack([data[start + 1: start+context_length+1] for start in starts])
        
        
        return (X, Y)








