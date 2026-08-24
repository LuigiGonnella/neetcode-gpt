import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        with torch.no_grad():
            stats = []
            for module in model:
                x = module(x)
                if isinstance(module, nn.ReLU):
                    if x.dim() >= 2: #batch dim exists
                        dead_frac = (x == 0).all(dim = 0).float().mean().item()
                    else:
                        dead_frac = (x == 0).float().mean().item()
                    
                    stats.append(round(dead_frac, 4))
        
        return stats

                    


    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        strictly_inc = True
        for (it, frac) in enumerate(dead_fractions):
            if strictly_inc and it and frac <= dead_fractions[it - 1]:
                strictly_inc = False
            if frac > 0.5:
                return 'use_leaky_relu'
            elif it == 0 and frac > 0.3:
                return 'reinitialize'
            
        
        if strictly_inc and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'
        
        return 'healthy'
            
            
