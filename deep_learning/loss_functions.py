import numpy as np
from mlfromscratch.utils import accuracy_score
from mlfromscratch.deep_learning.activation_functions import Sigmoid
from typing import Any


class Loss:
    """
    Base class for loss functions.

    Provides an interface for computing the loss and its gradient.

    Methods
    -------
    loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        Computes the loss between true and predicted values.
    gradient(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        Computes the gradient of the loss function with respect to the predictions.
    acc(y: np.ndarray, y_pred: np.ndarray) -> float:
        Computes the accuracy of the model.
    """
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the loss between true and predicted values.
        Should be implemented in the child class.
        """
        raise NotImplementedError()

    def gradient(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to predictions.
        Should be implemented in the child class.
        """
        raise NotImplementedError()

    def acc(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the accuracy of the model's predictions.
        By default, it returns 0. Can be overridden in specific loss functions.
        """
        return 0.0


class SquareLoss(Loss):
    """
    Mean Squared Error loss.

    This loss is typically used for regression tasks. It computes the squared difference
    between the predicted values and the true values, then averages it.

    Methods
    -------
    loss(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        Computes the mean squared error loss.
    gradient(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        Computes the gradient of the MSE loss.
    """

    def __init__(self):
        pass

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the Mean Squared Error loss.
        Args:
            y (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            np.ndarray: Computed loss value.
        """
        return 0.5 * np.power(y - y_pred, 2)

    def gradient(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the MSE loss with respect to predictions.
        Args:
            y (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            np.ndarray: Computed gradient.
        """
        return -(y - y_pred)


class CrossEntropy(Loss):
    """
    Cross-Entropy loss function.

    This loss is typically used for classification tasks. It computes the difference between
    the true label distribution and the predicted probability distribution.

    Methods
    -------
    loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
        Computes the cross-entropy loss between true labels and predicted probabilities.
    acc(y: np.ndarray, p: np.ndarray) -> float:
        Computes the accuracy of predictions.
    gradient(y: np.ndarray, p: np.ndarray) -> np.ndarray:
        Computes the gradient of the cross-entropy loss.
    """

    def __init__(self):
        pass

    def loss(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Computes the Cross-Entropy loss between true labels and predicted probabilities.
        Args:
            y (np.ndarray): True labels (one-hot encoded).
            p (np.ndarray): Predicted probabilities.
        Returns:
            np.ndarray: Computed cross-entropy loss.
        """
        # Avoid division by zero by clipping probabilities
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y: np.ndarray, p: np.ndarray) -> float:
        """
        Computes the accuracy of predictions by comparing the predicted class with the true class.
        Args:
            y (np.ndarray): True labels (one-hot encoded).
            p (np.ndarray): Predicted probabilities (predicted classes).
        Returns:
            float: Accuracy of the predictions.
        """
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the cross-entropy loss with respect to the predicted probabilities.
        Args:
            y (np.ndarray): True labels (one-hot encoded).
            p (np.ndarray): Predicted probabilities.
        Returns:
            np.ndarray: Computed gradient.
        """
        # Avoid division by zero by clipping probabilities
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y / p) + (1 - y) / (1 - p)
