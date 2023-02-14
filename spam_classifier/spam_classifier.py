from dataclasses import dataclass

import numpy as np

from .layer import Layer


@dataclass
class Configuration:
    layer_sizes: list
    activations: list
    learning_rate: float
    epochs: int


class SpamClassifier:
    def __init__(self, configuration: Configuration):
        self.layers = []
        for i in range(len(configuration.layer_sizes) - 1):
            layer = Layer(configuration.layer_sizes[i], configuration.layer_sizes[i + 1], configuration.activations[i])
            self.layers.append(layer)
        self.learning_rate = configuration.learning_rate

    def __forward(self, data):
        activations = data
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def train(self, x_train_data, y_train_data, epochs):
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
        return self.__forward(x_test_data)

