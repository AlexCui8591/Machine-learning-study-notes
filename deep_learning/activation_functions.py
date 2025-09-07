import numpy as np
from typing import Any

class Sigmoid:
    """
    Sigmoid activation function.

    Given an input x, the Sigmoid function is defined as:
        Sigmoid(x) = 1 / (1 + exp(-x))

    Attributes
    ----------
    None

    Methods
    -------
    __call__(x: np.ndarray) -> np.ndarray:
        Computes the Sigmoid activation for input x.
    gradient(x: np.ndarray) -> np.ndarray:
        Computes the gradient of the Sigmoid function w.r.t. input x.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the Sigmoid function."""
        sig_x = self.__call__(x)
        return sig_x * (1.0 - sig_x)


class Softmax:
    """
    Softmax activation function.

    Given an input x (usually a vector or 2D array where the last dimension
    represents classes), the Softmax function is defined as:
        Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Attributes
    ----------
    None

    Methods
    -------
    __call__(x: np.ndarray) -> np.ndarray:
        Computes the Softmax activation for input x.
    gradient(x: np.ndarray) -> np.ndarray:
        Computes a simplified gradient of the Softmax function w.r.t. x.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the Softmax activation function."""
        # Shift for numerical stability
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x_shifted)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute a simplified gradient for the Softmax function.

        Note: The gradient of Softmax is typically expressed as a Jacobian
              matrix. Here, we return p * (1 - p) for demonstration, which
              is analogous to the diagonal of the Jacobian for each class
              when used in cross-entropy settings.
        """
        p = self.__call__(x)
        return p * (1.0 - p)


class TanH:
    """
    Hyperbolic tangent (TanH) activation function.

    Given an input x, TanH is defined as:
        TanH(x) = 2 / (1 + exp(-2x)) - 1

    Attributes
    ----------
    None

    Methods
    -------
    __call__(x: np.ndarray) -> np.ndarray:
        Computes the TanH activation for input x.
    gradient(x: np.ndarray) -> np.ndarray:
        Computes the gradient of the TanH function w.r.t. input x.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the hyperbolic tangent function."""
        return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the TanH function."""
        tanh_x = self.__call__(x)
        return 1.0 - np.power(tanh_x, 2.0)


class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.

    Defined as:
        ReLU(x) = max(0, x)

    Attributes
    ----------
    None

    Methods
    -------
    __call__(x: np.ndarray) -> np.ndarray:
        Computes the ReLU activation for input x.
    gradient(x: np.ndarray) -> np.ndarray:
        Computes the gradient of the ReLU function w.r.t. input x.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the ReLU activation function."""
        return np.where(x >= 0, x, 0.0)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of ReLU."""
        return np.where(x >= 0, 1.0, 0.0)


class LeakyReLU:
    """
    Leaky ReLU activation function.

    Defined as:
        LeakyReLU(x) = x if x >= 0 else alpha * x

    Parameters
    ----------
    alpha : float
        Negative slope coefficient for x < 0.

    Attributes
    ----------
    alpha : float
        The slope for negative inputs.

    Methods
    -------
    __call__(x: np.ndarray) -> np.ndarray:
        Computes the Leaky ReLU activation for input x.
    gradient(x: np.ndarray) -> np.ndarray:
        Computes the gradient of the Leaky ReLU function w.r.t. input x.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the Leaky ReLU activation function."""
        return np.where(x >= 0.0, x, self.alpha * x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of Leaky ReLU."""
        return np.where(x >= 0.0, 1.0, self.alpha)


class ELU:
    """
    Exponential Linear Unit (ELU) activation function.

    Defined as:
        ELU(x) = x if x >= 0 else alpha * (exp(x) - 1)

    Parameters
    ----------
    alpha : float
        Scales the negative region.

    Attributes
    ----------
    alpha : float
        The scaling factor for negative inputs.

    Methods
    -------
    __call__(x: np.ndarray) -> np.ndarray:
        Computes the ELU activation for input x.
    gradient(x: np.ndarray) -> np.ndarray:
        Computes the gradient of the ELU function w.r.t. input x.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the ELU activation function."""
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1.0))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of ELU."""
        # gradient for x >= 0 is 1, otherwise (ELU(x) + alpha)
        return np.where(x >= 0.0, 1.0, self.__call__(x) + self.alpha)


class SELU:
    """
    Scaled Exponential Linear Unit (SELU).

    SELU(x) = scale * (x if x >= 0 else alpha * (exp(x) - 1))

    Attributes
    ----------
    alpha : float
        SELU alpha constant.
    scale : float
        SELU scale constant.

    References
    ----------
    - https://arxiv.org/abs/1706.02515
    - https://github.com/bioinf-jku/SNNs/blob/master/SelfNormalizingNetworks_MLP_MNIST.ipynb
    """

    def __init__(self):
        self.alpha = 1.6732632423543772
        self.scale = 1.0507009873554805

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the SELU activation function."""
        return self.scale * np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1.0))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of SELU."""
        return self.scale * np.where(x >= 0.0, 1.0, self.alpha * np.exp(x))


class SoftPlus:
    """
    SoftPlus activation function.

    Defined as:
        SoftPlus(x) = ln(1 + exp(x))

    Attributes
    ----------
    None

    Methods
    -------
    __call__(x: np.ndarray) -> np.ndarray:
        Computes the SoftPlus activation for input x.
    gradient(x: np.ndarray) -> np.ndarray:
        Computes the gradient of the SoftPlus function w.r.t. input x.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the SoftPlus activation function."""
        return np.log(1.0 + np.exp(x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the SoftPlus function."""
        return 1.0 / (1.0 + np.exp(-x))
