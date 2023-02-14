from dataclasses import dataclass
from typing import List

import numpy as np

from .layer import Layer, Activation


@dataclass
class Configuration:
    """
    This class defines the configuration of a neural network.

    :param layer_sizes: list of integers representing the size of each layer in the neural network.
                        For example, if `layer_sizes` is [2, 3, 1], it represents a neural network with
                        2 input neurons, 1 hidden layer with 3 neurons, and 1 output neuron.
    :param activations: list of enum strings representing the activation function for each layer in the neural network.
                        It should have the same length as `layer_sizes` minus 1.
                        For example, if `layer_sizes` is [2, 3, 1], and `activations` is [Activation.Sigmoid, Activation.ReLU],
                        it means that the activation function for the hidden layer is 'Sigmoid' and the activation
                        function for the output layer is 'ReLU'.
    :param learning_rate: float representing the learning rate used in the training process.
    :param epochs: int representing the number of epochs for training.
    """
    layer_sizes: list
    activations: List[Activation]
    learning_rate: float
    epochs: int


class SpamClassifier:
    """
    The SpamClassifier class provides a simple implementation of a neural network that can be used for spam classification.

    Args:
        configuration (Configuration): A Configuration object containing the configuration for the neural network.
    """
    def __init__(self, configuration: Configuration):
        self.layers = []
        for i in range(len(configuration.layer_sizes) - 1):
            layer = Layer(configuration.layer_sizes[i], configuration.layer_sizes[i + 1], configuration.activations[i])
            self.layers.append(layer)
        self.learning_rate = configuration.learning_rate

    def __forward(self, data):
        """
        Forward propagate the input through all the layers in the network.

        Args:
        data (np.ndarray): Input data to be propagated through the network.

        Returns:
        np.ndarray: Output activations after passing through all the layers in the network.
        """
        activations = data
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def train(self, x_train_data, y_train_data, epochs):
        """
        Train the spam classifier using the provided training data.

        Args:
            x_train_data (np.array): The input training data.
            y_train_data (np.array): The expected output training data.
            epochs (int): The number of training epochs to perform.

        """
        for epoch in range(epochs):
            activations = x_train_data
            for layer in self.layers:
                activations = layer.forward(activations)
            error = y_train_data - activations
            for i in range(len(self.layers) - 1, 0, -1):
                layer = self.layers[i]
                error = layer.backward(activations, error, self.learning_rate)
                activations = np.dot(error, layer.weights.T)

    def predict(self, x_test_data):
        """
        Make predictions for new, unseen data.

        Args:
            x_test_data (np.array): The input test data.

        Returns:
            np.array: The predictions for the given input data.

        """
        return self.__forward(x_test_data)

