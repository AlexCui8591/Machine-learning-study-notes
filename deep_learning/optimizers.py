import numpy as np
from typing import Any


class StochasticGradientDescent:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The learning rate for gradient updates.
    momentum : float, default=0
        The momentum factor to accelerate convergence.

    Methods
    -------
    update(w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        Updates the weights based on the gradient and momentum.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.w_updt = None

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        """
        Updates the weights using the gradient and momentum.

        Parameters
        ----------
        w : np.ndarray
            Current weights.
        grad_wrt_w : np.ndarray
            The gradient with respect to the weights.

        Returns
        -------
        np.ndarray
            Updated weights.
        """
        if self.w_updt is None:
            self.w_updt = np.zeros_like(w)
        # Update with momentum
        self.w_updt = self.momentum * self.w_updt + (1 - self.momentum) * grad_wrt_w
        return w - self.learning_rate * self.w_updt


class NesterovAcceleratedGradient:
    """
    Nesterov Accelerated Gradient (NAG) optimizer.

    Parameters
    ----------
    learning_rate : float, default=0.001
        The learning rate for gradient updates.
    momentum : float, default=0.4
        The momentum factor to accelerate convergence.

    Methods
    -------
    update(w: np.ndarray, grad_func: callable) -> np.ndarray:
        Updates the weights based on the gradient and momentum.
    """

    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.4) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.w_updt = np.array([])

    def update(self, w: np.ndarray, grad_func: callable) -> np.ndarray:
        """
        Updates the weights using Nesterov accelerated gradient.

        Parameters
        ----------
        w : np.ndarray
            Current weights.
        grad_func : callable
            A function that returns the gradient when called with weights.

        Returns
        -------
        np.ndarray
            Updated weights.
        """
        approx_future_grad = np.clip(grad_func(w - self.momentum * self.w_updt), -1, 1)
        if not self.w_updt.any():
            self.w_updt = np.zeros_like(w)

        self.w_updt = self.momentum * self.w_updt + self.learning_rate * approx_future_grad
        return w - self.w_updt


class Adagrad:
    """
    Adagrad optimizer.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The learning rate for gradient updates.

    Methods
    -------
    update(w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        Updates the weights based on the gradient and the accumulated squared gradients.
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.G = None  # Sum of squares of the gradients
        self.eps = 1e-8

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        """
        Updates the weights using Adagrad.

        Parameters
        ----------
        w : np.ndarray
            Current weights.
        grad_wrt_w : np.ndarray
            The gradient with respect to the weights.

        Returns
        -------
        np.ndarray
            Updated weights.
        """
        if self.G is None:
            self.G = np.zeros_like(w)

        self.G += np.power(grad_wrt_w, 2)
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.G + self.eps)


class Adadelta:
    """
    Adadelta optimizer.

    Parameters
    ----------
    rho : float, default=0.95
        The decay rate for the moving average of squared gradients.
    eps : float, default=1e-6
        A small constant to prevent division by zero.

    Methods
    -------
    update(w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        Updates the weights using Adadelta.
    """

    def __init__(self, rho: float = 0.95, eps: float = 1e-6) -> None:
        self.E_w_updt = None  # Running average of squared parameter updates
        self.E_grad = None  # Running average of the squared gradient of w
        self.w_updt = None  # Parameter update
        self.eps = eps
        self.rho = rho

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        """
        Updates the weights using Adadelta.

        Parameters
        ----------
        w : np.ndarray
            Current weights.
        grad_wrt_w : np.ndarray
            The gradient with respect to the weights.

        Returns
        -------
        np.ndarray
            Updated weights.
        """
        if self.w_updt is None:
            self.w_updt = np.zeros_like(w)
            self.E_w_updt = np.zeros_like(w)
            self.E_grad = np.zeros_like(grad_wrt_w)

        self.E_grad = self.rho * self.E_grad + (1 - self.rho) * np.power(grad_wrt_w, 2)

        RMS_delta_w = np.sqrt(self.E_w_updt + self.eps)
        RMS_grad = np.sqrt(self.E_grad + self.eps)

        adaptive_lr = RMS_delta_w / RMS_grad
        self.w_updt = adaptive_lr * grad_wrt_w

        self.E_w_updt = self.rho * self.E_w_updt + (1 - self.rho) * np.power(self.w_updt, 2)

        return w - self.w_updt


class RMSprop:
    """
    RMSprop optimizer.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The learning rate for gradient updates.
    rho : float, default=0.9
        The decay rate for the moving average of squared gradients.

    Methods
    -------
    update(w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        Updates the weights using RMSprop.
    """

    def __init__(self, learning_rate: float = 0.01, rho: float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.Eg = None  # Running average of the square gradients at w
        self.eps = 1e-8
        self.rho = rho

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        """
        Updates the weights using RMSprop.

        Parameters
        ----------
        w : np.ndarray
            Current weights.
        grad_wrt_w : np.ndarray
            The gradient with respect to the weights.

        Returns
        -------
        np.ndarray
            Updated weights.
        """
        if self.Eg is None:
            self.Eg = np.zeros_like(grad_wrt_w)

        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(grad_wrt_w, 2)
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.Eg + self.eps)


class Adam:
    """
    Adam optimizer.

    Parameters
    ----------
    learning_rate : float, default=0.001
        The learning rate for gradient updates.
    b1 : float, default=0.9
        The exponential decay rate for the first moment estimates.
    b2 : float, default=0.999
        The exponential decay rate for the second moment estimates.

    Methods
    -------
    update(w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        Updates the weights using Adam.
    """

    def __init__(self, learning_rate: float = 0.001, b1: float = 0.9, b2: float = 0.999) -> None:
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = None
        self.v = None
        self.b1 = b1
        self.b2 = b2

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        """
        Updates the weights using the Adam optimizer.

        Parameters
        ----------
        w : np.ndarray
            Current weights.
        grad_wrt_w : np.ndarray
            The gradient with respect to the weights.

        Returns
        -------
        np.ndarray
            Updated weights.
        """
        if self.m is None:
            self.m = np.zeros_like(grad_wrt_w)
            self.v = np.zeros_like(grad_wrt_w)

        self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        self.w_updt = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        return w - self.w_updt
