import numpy as np

from .layer import Layer


class SpamClassifier:
    def __init__(self, layers_config, activations, learning_rate):
        self.layers = []
        for i in range(len(layers_config) - 1):
            self.layers.append(Layer(layers_config[i], layers_config[i + 1], activations[i]))
        self.learning_rate = learning_rate

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

