import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        logits = np.matmul(X, weights)

        def softmax(z):
            shifted = z - max(z)
            exp = np.exp(shifted)
            probs = exp / np.sum(exp)
            return np.round(probs, 5)

        return np.round(logits, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        mse = np.mean(np.square(model_prediction - ground_truth))
        return np.round(mse, 5)
