import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        n_features = X.shape[1]
        n = X.shape[0]
        w = np.zeros(n_features, dtype=float)
        b = 0.0

        for epoch in range(epochs):
            y_hat = X @ w.T + b
         
            error = y_hat - y
            loss = np.mean(np.square(error))

            err1 = error.reshape(1, -1) 
            dw = ((2 / n) * err1 @ X).squeeze()
            db = (2 / n) * np.sum(error)

            w = w - lr * dw
            b = b - lr * db

        return (np.round(w, 5), np.round(b, 5))







