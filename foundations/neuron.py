import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        
        logits = np.dot(x, w) + b

        def get_act(activation, logits):

            if activation == "relu":
                return max(0.0, logits)
            elif activation == "sigmoid":
                return 1.0 / (1.0 + np.exp(-logits))
            else:
                return logits
        
        act = get_act(activation, logits)
        return np.round(act, 5)


