import numpy as np


class SpamClassifier:
    def __init__(self, hidden_layer_size=15, learning_rate=0.05, epochs=12600, reg_lambda=0.01):
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.weights_input_to_hidden = None
        self.weights_hidden_to_output = None

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def train(self, data, features):
        n_samples, n_features = data.shape

        limit = np.sqrt(6 / (n_features + self.hidden_layer_size))
        self.weights_input_to_hidden = np.random.uniform(-limit, limit, size=(n_features, self.hidden_layer_size))
        self.weights_hidden_to_output = np.random.uniform(-limit, limit, size=(self.hidden_layer_size, 1))

        for i in range(self.epochs):
            # forward pass
            hidden_layer = SpamClassifier.__sigmoid(np.dot(data, self.weights_input_to_hidden))
            output = SpamClassifier.__sigmoid(np.dot(hidden_layer, self.weights_hidden_to_output))

            # loss calculation
            loss = (-1/n_samples) * np.sum(features * np.log(output) + (1 - features) * np.log(1 - output))
            # add L2 regularization penalty to loss
            L2_penalty = (self.reg_lambda/(2*n_samples)) * (np.sum(np.square(self.weights_input_to_hidden)) + np.sum(np.square(self.weights_hidden_to_output)))
            loss += L2_penalty

            # backward pass
            error = features.reshape(-1, 1) - output
            d_output = error * output * (1 - output)
            error_hidden_layer = np.dot(d_output, self.weights_hidden_to_output.T)
            d_hidden_layer = error_hidden_layer * hidden_layer * (1 - hidden_layer)

            # update weights with L2 regularization penalty
            self.weights_hidden_to_output += self.learning_rate * (np.dot(hidden_layer.T, d_output) - (self.reg_lambda/n_samples) * self.weights_hidden_to_output)
            self.weights_input_to_hidden += self.learning_rate * (np.dot(data.T, d_hidden_layer) - (self.reg_lambda/n_samples) * self.weights_input_to_hidden)

    def predict(self, data):
        hidden_layer = SpamClassifier.__sigmoid(np.dot(data, self.weights_input_to_hidden))
        output = SpamClassifier.__sigmoid(np.dot(hidden_layer, self.weights_hidden_to_output))
        y_pred = (output > 0.5).astype(int).flatten()
        return y_pred
