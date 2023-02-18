import numpy as np


class SpamClassifier:
    def __init__(self, hidden_layer_size=13, learning_rate=0.01, epochs=1000):
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_input_to_hidden = None
        self.weights_hidden_to_output = None

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def train(self, data, features):
        n_samples, n_features = data.shape
        self.weights_input_to_hidden = np.random.randn(n_features, self.hidden_layer_size) / np.sqrt(n_features)
        self.weights_hidden_to_output = np.random.randn(self.hidden_layer_size, 1) / np.sqrt(self.hidden_layer_size)

        for _ in range(self.epochs):
            # forward pass
            hidden_layer = SpamClassifier.__sigmoid(np.dot(data, self.weights_input_to_hidden))
            output = SpamClassifier.__sigmoid(np.dot(hidden_layer, self.weights_hidden_to_output))

            # backward pass
            error = features.reshape(-1, 1) - output
            d_output = error * output * (1 - output)
            error_hidden_layer = np.dot(d_output, self.weights_hidden_to_output.T)
            d_hidden_layer = error_hidden_layer * hidden_layer * (1 - hidden_layer)

            # update weights
            self.weights_hidden_to_output += self.learning_rate * np.dot(hidden_layer.T, d_output)
            self.weights_input_to_hidden += self.learning_rate * np.dot(data.T, d_hidden_layer)

    def predict(self, data):
        hidden_layer = SpamClassifier.__sigmoid(np.dot(data, self.weights_input_to_hidden))
        output = SpamClassifier.__sigmoid(np.dot(hidden_layer, self.weights_hidden_to_output))
        y_pred = (output > 0.5).astype(int).flatten()
        return y_pred