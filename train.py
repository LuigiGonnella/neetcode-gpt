import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        cross_entropy = nn.CrossEntropyLoss()
        n = len(data)


        for epoch in range(epochs):
            torch.manual_seed(epoch)

            #sample a random batch with X, Y
            starts = torch.randint(n - context_length, size = (batch_size,))
            X = torch.stack([data[start: start + context_length] for start in starts])
            Y = torch.stack([data[start + 1: start + context_length + 1] for start in starts]) #(B, C, ) --> CrossEntropy requires logits per target in pred: (B, C, vocab_size) and just target ids for Y: (B, C, )

            optimizer.zero_grad()

            logits = model(X)
            logits = logits.reshape((logits.shape[0] * logits.shape[1], logits.shape[2]))
            loss = cross_entropy(logits, Y.reshape((Y.shape[0] * Y.shape[1],)))

            loss.backward()
            optimizer.step()
        
        return round(loss.item(), 4)



