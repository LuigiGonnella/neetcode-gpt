import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.fcl1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.2) 
        self.fcl2 = nn.Linear(512, 10)
        self.sig = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x1 = self.dp1(self.relu(self.fcl1(images)))
        x2 = self.fcl2(x1)

        return torch.round(self.sig(x2), decimals = 4)
