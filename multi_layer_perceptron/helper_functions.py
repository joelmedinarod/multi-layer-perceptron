import numpy as np


def accuracy_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates classification accuracy between truth labels and predictions.

    Arguments:
        y_true: Truth labels for predictions.
        y_pred: Predictions to be compared to predictions.

    Returns:
    accuracy: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = len(y_true[y_true == y_pred])
    return (correct / len(y_pred)) * 100


def ReLU(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation function"""
    return np.maximum(0, x)


def ReLU_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the ReLU activation function"""
    return np.where(x > 0, 1, 0)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function

    Arguments:
    x: Logits

    Returns:
    pred_probs: prediction probabilities
    """
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shifted_x)
    sum_exps = np.sum(exps, axis=1, keepdims=True)
    return exps / sum_exps


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Cross Entropy Loss Function"""
    n_samples = y_true.shape[0]
    res = y_pred[np.arange(n_samples), y_true]
    return -np.mean(np.log(res + 1e-15))


def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Derivative of the Cross Entropy Loss Function"""
    n_samples = y_true.shape[0]
    grad = y_pred.copy()
    grad[np.arange(n_samples), y_true] -= 1
    return grad / n_samples


def xavier_initialization(num_inputs: int, num_outputs: int) -> np.ndarray:
    """Xavier initialization for initializing weights of a linear layer"""
    return np.random.randn(num_inputs, num_outputs) * np.sqrt(
        2.0 / (num_inputs + num_outputs)
    )
