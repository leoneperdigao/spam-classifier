import numpy as np
from enum import Enum


class Activation(Enum):
    """
    Enum class representing activation functions used in the neural network.

    Attributes:
    SIGMOID : str
        String representation of sigmoid activation function.
    RELU : str
        String representation of ReLU activation function.
    """
    SIGMOID = 'sigmoid'
    RELU = 'relu'


class Layer:
    """
    Class representing a single layer of the neural network.

    Parameters:
    input_size (int): Number of neurons in the input layer.
    output_size (int): Number of neurons in the output layer.
    activation_fn (Activation, optional): Activation function to be used in the layer. Defaults to Activation.RELU.
    scaling_factor (float, optional): Scaling factor to be applied to the weights. Defaults to 0.01.

    Attributes:
    weights (numpy.ndarray): Matrix of weights for the layer.
    biases (numpy.ndarray): Vector of biases for the layer.
    activation_fn (Activation): Activation function used in the layer.
    """
    def __init__(self, input_size, output_size, activation_fn: Activation = Activation.RELU, scaling_factor: float = 0.01):
        self.weights = np.random.randn(input_size, output_size) * scaling_factor
        self.biases = np.zeros((1, output_size))
        self.activation_fn = activation_fn

    def forward(self, data):
        """
        Compute the forward pass through the layer for a given input.

        Parameters:
        data (numpy.ndarray): Input data for the layer.

        Returns:
        numpy.ndarray: Output activations of the layer.
        """
        if self.activation_fn == Activation.SIGMOID:
            return 1 / (1 + np.exp(-(np.dot(data, self.weights) + self.biases)))
        elif self.activation_fn == Activation.RELU:
            return np.maximum(0, np.dot(data, self.weights) + self.biases)

        raise ValueError('Invalid activation function')

    def backward(self, input_data, delta, learning_rate):
        """
        Compute the backward pass through the layer and update the weights and biases.

        Parameters:
        input_data (numpy.ndarray): Input activations for the layer.
        delta (numpy.ndarray): Gradient of the loss with respect to the layer output.
        learning_rate (float): Learning rate for the update step.

        Returns:
        numpy.ndarray: Gradient of the loss with respect to the layer input.
        """
        if self.activation_fn == Activation.SIGMOID:
            delta = delta * input_data * (1 - input_data)
        elif self.activation_fn == Activation.RELU:
            delta = delta * (input_data > 0)

        self.weights += learning_rate * np.dot(input_data.T, delta)
        self.biases += learning_rate * np.sum(delta, axis=0, keepdims=True)

        return delta
