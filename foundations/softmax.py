import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        # max_el = max(z)
        # den = sum([np.exp(el - max_el) for el in z])
        # res = [np.exp(el - max_el)/den for el in z]

        shifted = z - np.max(z)
        exps = np.exp(shifted) #array of e^(z - max(z))
        res = exps / np.sum(exps)
        return np.round(res, 4)
