import numpy as np
from enum import Enum


class Activation(Enum):
    SIGMOID = 'sigmoid'
    RELU = 'relu'


class Layer:
    def __init__(self, input_size, output_size, activation_fn: Activation = Activation.RELU, scaling_factor: float = 0.01):
        self.weights = np.random.randn(input_size, output_size) * scaling_factor
        self.biases = np.zeros((1, output_size))
        self.activation_fn = activation_fn

    def forward(self, data):
        if self.activation_fn == Activation.SIGMOID:
            return 1 / (1 + np.exp(-(np.dot(data, self.weights) + self.biases)))
        elif self.activation_fn == Activation.RELU:
            return np.maximum(0, np.dot(data, self.weights) + self.biases)

        raise ValueError('Invalid activation function')

    def backward(self, input_data, delta, learning_rate):
        if self.activation_fn == Activation.SIGMOID:
            delta = delta * input_data * (1 - input_data)
        elif self.activation_fn == Activation.RELU:
            delta = delta * (input_data > 0)

        self.weights += learning_rate * np.dot(input_data.T, delta)
        self.biases += learning_rate * np.sum(delta, axis=0, keepdims=True)

        return delta
