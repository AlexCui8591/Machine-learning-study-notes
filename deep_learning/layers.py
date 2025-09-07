import math
import copy
import numpy as np

from mlfromscratch.deep_learning.activation_functions import (
    Sigmoid, ReLU, SoftPlus, LeakyReLU,
    TanH, ELU, SELU, Softmax
)

# Activation function lookup
activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'selu': SELU,
    'elu': ELU,
    'softmax': Softmax,
    'leaky_relu': LeakyReLU,
    'tanh': TanH,
    'softplus': SoftPlus
}


class Layer:
    """
    Base class for all neural network layers.
    """

    def set_input_shape(self, shape: tuple) -> None:
        """
        Sets the shape that the layer expects of the input in forward_pass.
        """
        self.input_shape = shape

    def layer_name(self) -> str:
        """
        Returns the name of the layer. Used in model summary.
        """
        return self.__class__.__name__

    def parameters(self) -> int:
        """
        Returns the number of trainable parameters used by the layer.
        Default is 0 for layers without parameters.
        """
        return 0

    def forward_pass(self, X: np.ndarray, training: bool) -> np.ndarray:
        """
        Forward pass logic. Propagates the signal forward in the network.
        """
        raise NotImplementedError()

    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass logic. Propagates the accumulated gradient backward.
        If the layer has trainable weights, these should be updated here.
        """
        raise NotImplementedError()

    def output_shape(self) -> tuple:
        """
        Returns the shape of the output produced by forward_pass.
        """
        raise NotImplementedError()


class Dense(Layer):
    """
    A fully-connected (Dense) neural network layer.

    Parameters
    ----------
    n_units : int
        The number of neurons in the layer.
    input_shape : tuple, optional
        Shape of the expected input. For Dense layers, this is a single int specifying
        the input features. Must be set if this is the first layer in the network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        self.layer_input: np.ndarray = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W: np.ndarray = None
        self.w0: np.ndarray = None

    def initialize(self, optimizer) -> None:
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self) -> int:
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        W = self.W.copy()

        if self.trainable:
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)

            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        accum_grad = accum_grad.dot(W.T)
        return accum_grad

    def output_shape(self) -> tuple:
        return (self.n_units, )


class RNN(Layer):
    """
    A Vanilla Fully-Connected Recurrent Neural Network (RNN) layer.

    Parameters
    ----------
    n_units : int
        Number of hidden states in the layer.
    activation : str
        Name of the activation function to apply at each timestep.
    bptt_trunc : int
        Number of timesteps through which to backpropagate.
    input_shape : tuple
        Expected input shape (timesteps, input_dim).
        Must be set if this layer is first in the network.

    Reference
    ---------
    http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
    """

    def __init__(self, n_units: int, activation: str = 'tanh',
                 bptt_trunc: int = 5, input_shape: tuple = None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation = activation_functions[activation]()
        self.trainable = True
        self.bptt_trunc = bptt_trunc
        self.W: np.ndarray = None
        self.V: np.ndarray = None
        self.U: np.ndarray = None

    def initialize(self, optimizer) -> None:
        timesteps, input_dim = self.input_shape
        limit = 1 / math.sqrt(input_dim)
        self.U = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        limit = 1 / math.sqrt(self.n_units)
        self.V = np.random.uniform(-limit, limit, (input_dim, self.n_units))
        self.W = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.U_opt = copy.copy(optimizer)
        self.V_opt = copy.copy(optimizer)
        self.W_opt = copy.copy(optimizer)

    def parameters(self) -> int:
        return np.prod(self.W.shape) + np.prod(self.U.shape) + np.prod(self.V.shape)

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.layer_input = X
        batch_size, timesteps, input_dim = X.shape

        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps + 1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))

        self.states[:, -1] = np.zeros((batch_size, self.n_units))
        for t in range(timesteps):
            self.state_input[:, t] = X[:, t].dot(self.U.T) + self.states[:, t - 1].dot(self.W.T)
            self.states[:, t] = self.activation(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.V.T)

        return self.outputs

    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        _, timesteps, _ = accum_grad.shape
        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)
        accum_grad_next = np.zeros_like(accum_grad)

        for t in reversed(range(timesteps)):
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])
            grad_wrt_state = accum_grad[:, t].dot(self.V) * self.activation.gradient(self.state_input[:, t])
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
            for t_ in reversed(range(max(0, t - self.bptt_trunc), t + 1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_ - 1])
                grad_wrt_state = grad_wrt_state.dot(self.W) * self.activation.gradient(self.state_input[:, t_ - 1])

        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)
        return accum_grad_next

    def output_shape(self) -> tuple:
        return self.input_shape


class Conv2D(Layer):
    """
    A 2D Convolution Layer.

    Parameters
    ----------
    n_filters : int
        Number of filters for the convolution.
    filter_shape : tuple
        (filter_height, filter_width)
    input_shape : tuple, optional
        Shape of the input as (channels, height, width). Must be set if first layer in the network.
    padding : str, default='same'
        Either 'same' or 'valid'. 'same' preserves height/width, 'valid' adds no padding.
    stride : int, default=1
        Stride length for the filters.
    """

    def __init__(self, n_filters: int, filter_shape: tuple,
                 input_shape: tuple = None,
                 padding: str = 'same',
                 stride: int = 1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape
        self.trainable = True
        self.W: np.ndarray = None
        self.w0: np.ndarray = None

    def initialize(self, optimizer) -> None:
        filter_height, filter_width = self.filter_shape
        channels = self.input_shape[0]
        limit = 1 / math.sqrt(np.prod(self.filter_shape))
        self.W = np.random.uniform(-limit, limit,
                                   size=(self.n_filters, channels, filter_height, filter_width))
        self.w0 = np.zeros((self.n_filters, 1))
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self) -> int:
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        batch_size, channels, height, width = X.shape
        self.layer_input = X
        self.X_col = image_to_column(X, self.filter_shape, stride=self.stride, output_shape=self.padding)
        self.W_col = self.W.reshape((self.n_filters, -1))
        output = self.W_col.dot(self.X_col) + self.w0
        output = output.reshape(self.output_shape() + (batch_size,))
        return output.transpose(3, 0, 1, 2)

    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        accum_grad = accum_grad.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)
        if self.trainable:
            grad_w = accum_grad.dot(self.X_col.T).reshape(self.W.shape)
            grad_w0 = np.sum(accum_grad, axis=1, keepdims=True)

            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        accum_grad = self.W_col.T.dot(accum_grad)
        accum_grad = column_to_image(accum_grad,
                                     self.layer_input.shape,
                                     self.filter_shape,
                                     stride=self.stride,
                                     output_shape=self.padding)
        return accum_grad

    def output_shape(self) -> tuple:
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        out_height = (height + sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        out_width = (width + sum(pad_w) - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(out_height), int(out_width)


class BatchNormalization(Layer):
    """
    Batch Normalization layer.

    Parameters
    ----------
    momentum : float, default=0.99
        Momentum for the moving average of mean and variance.
    """

    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.trainable = True
        self.eps = 0.01
        self.running_mean: np.ndarray = None
        self.running_var: np.ndarray = None
        self.gamma: np.ndarray = None
        self.beta: np.ndarray = None

    def initialize(self, optimizer) -> None:
        self.gamma = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)
        self.gamma_opt = copy.copy(optimizer)
        self.beta_opt = copy.copy(optimizer)

    def parameters(self) -> int:
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if self.running_mean is None:
            self.running_mean = np.mean(X, axis=0)
            self.running_var = np.var(X, axis=0)

        if training and self.trainable:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        self.X_centered = X - mean
        self.stddev_inv = 1 / np.sqrt(var + self.eps)
        X_norm = self.X_centered * self.stddev_inv
        output = self.gamma * X_norm + self.beta
        return output

    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        gamma = self.gamma
        if self.trainable:
            X_norm = self.X_centered * self.stddev_inv
            grad_gamma = np.sum(accum_grad * X_norm, axis=0)
            grad_beta = np.sum(accum_grad, axis=0)
            self.gamma = self.gamma_opt.update(self.gamma, grad_gamma)
            self.beta = self.beta_opt.update(self.beta, grad_beta)

        batch_size = accum_grad.shape[0]
        accum_grad = (1 / batch_size) * gamma * self.stddev_inv * (
            batch_size * accum_grad
            - np.sum(accum_grad, axis=0)
            - self.X_centered * (self.stddev_inv**2) * np.sum(accum_grad * self.X_centered, axis=0)
        )
        return accum_grad

    def output_shape(self) -> tuple:
        return self.input_shape


class PoolingLayer(Layer):
    """
    A parent class for pooling layers (MaxPooling2D, AveragePooling2D).

    Parameters
    ----------
    pool_shape : tuple, default=(2, 2)
        Shape of the pooling window (height, width).
    stride : int, default=1
        The stride with which the pooling window is moved.
    padding : int, default=0
        If > 0, zero-padding is added to the input.
    """

    def __init__(self, pool_shape: tuple = (2, 2), stride: int = 1, padding: int = 0):
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding
        self.trainable = True
        self.input_shape: tuple = None
        self.layer_input: np.ndarray = None

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.layer_input = X
        batch_size, channels, height, width = X.shape
        _, out_height, out_width = self.output_shape()
        X = X.reshape(batch_size * channels, 1, height, width)
        X_col = image_to_column(X, self.pool_shape, self.stride, self.padding)
        output = self._pool_forward(X_col)
        output = output.reshape(out_height, out_width, batch_size, channels)
        output = output.transpose(2, 3, 0, 1)
        return output

    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        batch_size, _, _, _ = accum_grad.shape
        channels, height, width = self.input_shape
        accum_grad = accum_grad.transpose(2, 3, 0, 1).ravel()
        accum_grad_col = self._pool_backward(accum_grad)
        accum_grad = column_to_image(
            accum_grad_col,
            (batch_size * channels, 1, height, width),
            self.pool_shape,
            self.stride,
            0
        )
        accum_grad = accum_grad.reshape((batch_size,) + self.input_shape)
        return accum_grad

    def _pool_forward(self, X_col: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _pool_backward(self, accum_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def output_shape(self) -> tuple:
        channels, height, width = self.input_shape
        out_height = (height - self.pool_shape[0]) / self.stride + 1
        out_width = (width - self.pool_shape[1]) / self.stride + 1
        assert out_height % 1 == 0, "Invalid pooling parameters."
        assert out_width % 1 == 0, "Invalid pooling parameters."
        return channels, int(out_height), int(out_width)


class MaxPooling2D(PoolingLayer):
    """
    2D Max Pooling layer.
    """

    def _pool_forward(self, X_col: np.ndarray) -> np.ndarray:
        arg_max = np.argmax(X_col, axis=0).flatten()
        output = X_col[arg_max, range(arg_max.size)]
        self.cache = arg_max
        return output

    def _pool_backward(self, accum_grad: np.ndarray) -> np.ndarray:
        accum_grad_col = np.zeros((np.prod(self.pool_shape), accum_grad.size))
        arg_max = self.cache
        accum_grad_col[arg_max, range(accum_grad.size)] = accum_grad
        return accum_grad_col


class AveragePooling2D(PoolingLayer):
    """
    2D Average Pooling layer.
    """

    def _pool_forward(self, X_col: np.ndarray) -> np.ndarray:
        return np.mean(X_col, axis=0)

    def _pool_backward(self, accum_grad: np.ndarray) -> np.ndarray:
        accum_grad_col = np.zeros((np.prod(self.pool_shape), accum_grad.size))
        accum_grad_col[:, range(accum_grad.size)] = (1.0 / accum_grad_col.shape[0]) * accum_grad
        return accum_grad_col


class ConstantPadding2D(Layer):
    """
    Adds constant padding to the input.

    Parameters
    ----------
    padding : tuple
        Amount of padding (pad_h, pad_w) or
        ((pad_h0, pad_h1), (pad_w0, pad_w1)).
    padding_value : int, default=0
        The constant value to pad with.
    """

    def __init__(self, padding: tuple, padding_value: int = 0):
        self.trainable = True
        if not isinstance(padding[0], tuple):
            padding = ((padding[0], padding[0]), padding[1])
        if not isinstance(padding[1], tuple):
            padding = (padding[0], (padding[1], padding[1]))
        self.padding = padding
        self.padding_value = padding_value
        self.input_shape: tuple = None

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        return np.pad(
            X,
            pad_width=((0, 0), (0, 0), self.padding[0], self.padding[1]),
            mode="constant",
            constant_values=self.padding_value
        )

    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        pad_top, pad_left = self.padding[0][0], self.padding[1][0]
        height, width = self.input_shape[1], self.input_shape[2]
        return accum_grad[:, :, pad_top:pad_top + height, pad_left:pad_left + width]

    def output_shape(self) -> tuple:
        new_height = self.input_shape[1] + sum(self.padding[0])
        new_width = self.input_shape[2] + sum(self.padding[1])
        return (self.input_shape[0], new_height, new_width)


class ZeroPadding2D(ConstantPadding2D):
    """
    Zero-padding layer for 2D inputs.

    Parameters
    ----------
    padding : tuple
        Same usage as ConstantPadding2D, except the constant padding value is always 0.
    """

    def __init__(self, padding: tuple):
        if isinstance(padding[0], int):
            padding = ((padding[0], padding[0]), padding[1])
        if isinstance(padding[1], int):
            padding = (padding[0], (padding[1], padding[1]))
        super().__init__(padding=padding, padding_value=0)


class Flatten(Layer):
    """
    Flattens a multi-dimensional input into a 2D array of shape (batch_size, -1).

    Parameters
    ----------
    input_shape : tuple, optional
        Shape of the input (batch_size, ...). Must be set if first layer.
    """

    def __init__(self, input_shape: tuple = None):
        self.trainable = True
        self.input_shape = input_shape
        self.prev_shape: tuple = None

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        return accum_grad.reshape(self.prev_shape)

    def output_shape(self) -> tuple:
        return (np.prod(self.input_shape),)


class UpSampling2D(Layer):
    """
    Nearest neighbor up sampling for 2D inputs.
    Repeats rows/columns of the data by size[0] and size[1].

    Parameters
    ----------
    size : tuple, default=(2,2)
        (size_y, size_x) - the number of times each axis is repeated.
    input_shape : tuple, optional
        (channels, height, width). Must be set if this is the first layer.
    """

    def __init__(self, size: tuple = (2, 2), input_shape: tuple = None):
        self.trainable = True
        self.size = size
        self.input_shape = input_shape
        self.prev_shape: tuple = None

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.prev_shape = X.shape
        return X.repeat(self.size[0], axis=2).repeat(self.size[1], axis=3)

    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        return accum_grad[:, :, ::self.size[0], ::self.size[1]]

    def output_shape(self) -> tuple:
        channels, height, width = self.input_shape
        return channels, self.size[0] * height, self.size[1] * width


class Reshape(Layer):
    """
    Reshapes the input tensor into a specified shape.

    Parameters
    ----------
    shape : tuple
        The target shape (excluding batch_size).
    input_shape : tuple, optional
        Must be set if this is the first layer in the network.
    """

    def __init__(self, shape: tuple, input_shape: tuple = None):
        self.trainable = True
        self.shape = shape
        self.input_shape = input_shape
        self.prev_shape: tuple = None

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.prev_shape = X.shape
        return X.reshape((X.shape[0],) + self.shape)

    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        return accum_grad.reshape(self.prev_shape)

    def output_shape(self) -> tuple:
        return self.shape


class Dropout(Layer):
    """
    A layer that randomly sets a fraction p of input units to zero.

    Parameters
    ----------
    p : float, default=0.2
        Probability of setting a unit to zero.
    """

    def __init__(self, p: float = 0.2):
        self.p = p
        self._mask: np.ndarray = None
        self.input_shape: tuple = None
        self.n_units: int = None
        self.pass_through = True
        self.trainable = True

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        c = (1 - self.p)
        if training:
            self._mask = np.random.rand(*X.shape) > self.p
            return X * self._mask
        return X * c

    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        return accum_grad * self._mask

    def output_shape(self) -> tuple:
        return self.input_shape


class Activation(Layer):
    """
    A layer that applies an activation function to the input.

    Parameters
    ----------
    name : str
        Name of the activation function.
    """

    def __init__(self, name: str):
        self.activation_name = name
        self.activation_func = activation_functions[name]()
        self.trainable = True
        self.input_shape: tuple = None
        self.layer_input: np.ndarray = None

    def layer_name(self) -> str:
        return f"Activation ({self.activation_func.__class__.__name__})"

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.layer_input = X
        return self.activation_func(X)

    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        return accum_grad * self.activation_func.gradient(self.layer_input)

    def output_shape(self) -> tuple:
        return self.input_shape


def determine_padding(filter_shape: tuple, output_shape: str = "same") -> tuple:
    """
    Calculates the padding required for 'same' or 'valid' convolution.

    Parameters
    ----------
    filter_shape : tuple
        (filter_height, filter_width)
    output_shape : str, default='same'
        Either 'same' or 'valid'.

    Returns
    -------
    tuple of tuples
        ((pad_h1, pad_h2), (pad_w1, pad_w2))
    """
    if output_shape == "valid":
        return (0, 0), (0, 0)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape
        pad_h1 = int(math.floor((filter_height - 1) / 2))
        pad_h2 = int(math.ceil((filter_height - 1) / 2))
        pad_w1 = int(math.floor((filter_width - 1) / 2))
        pad_w2 = int(math.ceil((filter_width - 1) / 2))
        return (pad_h1, pad_h2), (pad_w1, pad_w2)


def get_im2col_indices(
    images_shape: tuple,
    filter_shape: tuple,
    padding: tuple,
    stride: int = 1
) -> tuple:
    """
    Computes the indices needed to transform images into columns (im2col).
    Reference: CS231n Stanford

    Parameters
    ----------
    images_shape : tuple
        (batch_size, channels, height, width)
    filter_shape : tuple
        (filter_height, filter_width)
    padding : tuple
        (pad_h, pad_w) from determine_padding
    stride : int, default=1

    Returns
    -------
    tuple
        (k, i, j) indices for indexing into padded images.
    """
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + sum(pad_h) - filter_height) / stride + 1)
    out_width = int((width + sum(pad_w) - filter_width) / stride + 1)

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)
    return k, i, j


def image_to_column(
    images: np.ndarray,
    filter_shape: tuple,
    stride: int,
    output_shape: str = 'same'
) -> np.ndarray:
    """
    Transforms image-shaped input to column shape (im2col).

    Reference: CS231n Stanford

    Parameters
    ----------
    images : np.ndarray
        Shape (batch_size, channels, height, width)
    filter_shape : tuple
        (filter_height, filter_width)
    stride : int
        Convolution stride
    output_shape : str, default='same'

    Returns
    -------
    np.ndarray
        Column-shaped representation of the image patches.
    """
    filter_height, filter_width = filter_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')
    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    return cols


def column_to_image(
    cols: np.ndarray,
    images_shape: tuple,
    filter_shape: tuple,
    stride: int,
    output_shape: str = 'same'
) -> np.ndarray:
    """
    Transforms column-shaped input back to image shape (col2im).

    Reference: CS231n Stanford

    Parameters
    ----------
    cols : np.ndarray
        Column-shaped data.
    images_shape : tuple
        (batch_size, channels, height, width)
    filter_shape : tuple
        (filter_height, filter_width)
    stride : int
        Convolution stride
    output_shape : str, default='same'

    Returns
    -------
    np.ndarray
        Images reshaped from columns with optional padding removed.
    """
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    height_padded = height + sum(pad_h)
    width_padded = width + sum(pad_w)
    images_padded = np.zeros((batch_size, channels, height_padded, width_padded), dtype=cols.dtype)

    k, i, j = get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w), stride)
    cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
    cols = cols.transpose(2, 0, 1)
    np.add.at(images_padded, (slice(None), k, i, j), cols)
    return images_padded[:, :, pad_h[0]:height + pad_h[0], pad_w[0]:width + pad_w[0]]
