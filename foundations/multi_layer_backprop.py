import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        ret = {}
        N = len(y_true)
        
        z1 = x @ W1.T + b1
        a1 = np.maximum(z1, 0)

        z2 = a1 @ W2.T + b2
        err = z2 - y_true

        loss = np.round(np.mean(np.square(err)), 4)

        relu_der = [0 if el <= 0 else 1 for el in z1]

        dz2 = (2 / N) * err

        dW2 = np.round(dz2.reshape(-1, 1) @ a1.reshape(1, -1), 4) #column M * row N gives (M, N) --> OUTER PRODUCT
        db2 = np.round(dz2, 4)

        da1 = dz2.reshape(1, -1) @ W2
        da1 = da1.flatten()

        dz1 = da1 * relu_der

        dW1 = np.round( dz1.reshape(-1, 1) @ x.reshape(1, -1), 4)
        db1 = np.round(dz1, 4)

        ret['loss'] = loss
        ret['dW2'] = dW2
        ret['dW1'] = dW1
        ret['db1'] = db1
        ret['db2'] = db2

        return ret

        