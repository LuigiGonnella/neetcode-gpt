import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad(): #avoid to build computational graph, doesn't keep track of gradients
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    mean_val = round(x.mean().item(), 4)
                    std_val = round(x.std().item(), 4)
                    if x.dim() >= 2: #we ensure that x has (batch_dim, neurons) where neurons is the number of output features
                        dead_fraction = round((x <= 0).all(dim = 0) #x <= 0 gets a bool matrix (batch_dim, neurons), while all gets a bool array, one per column (dim = 0, col 0 is neuron/feature 0 across all batches, ...), True if all values are True in the row and False otherwise --> if True means all neurons od a row (a batch sample) were <= 0
                        .float().mean().item(), 4)#float trasforms Flase in 0.0 and True in 1.0 and .mean gets the fraction
                    else:
                        dead_fraction = round((x <= 0).float().mean().item(), 4)
                    stats.append({'mean': mean_val, 'std': std_val, 'dead_fraction': dead_fraction})
        
        return stats


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        
        model.zero_grad() #clears gradients per weight
        out = model(x)
        loss = nn.MSELoss()(out, y)
        loss.backward() #computes gradients
        stats = []

        for module in model.children():
            if isinstance(module, nn.Linear):

                grad = module.weight.grad #(in_feat, out_feat) matrix with gradients per weight
                mean_val = round(grad.mean().item(), 4) #global, across all weights of all neurons (if we wanted mean per neuron(row) we would have done mean(dim=1))
                std_val = round(grad.std().item(), 4)
                norm_val = round(torch.norm(grad).item(), 4)
                stats.append({'mean': mean_val, 'std': std_val, 'norm': norm_val})
            
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for layer in activation_stats:
            if layer['dead_fraction'] > 0.5:
                return 'dead_neurons'
            elif layer['std'] < 0.1:
                return 'vanishing_gradients'
            elif layer['std'] > 10.0:
                return 'exploding_gradients'
        
        for layer in gradient_stats:
            if layer['norm'] > 1000:
                return 'exploding_gradients'
            elif layer['norm'] < 1e-5:
                return 'vanishing_gradients'
        
        return 'healthy'

