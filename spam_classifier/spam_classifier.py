import numpy as np


class Layer:
    def __init__(self, n_input, n_output, activation_func):
        self.last_output = None
        self.last_input = None
        variance = 2.0 / (n_input + n_output)
        mean = 0
        fan_in = n_input

        np.random.seed(0)  # set a random seed for reproducibility

        self.weights = np.random.normal(loc=mean, scale=np.sqrt(variance/fan_in), size=(n_input, n_output))
        self.biases = np.zeros((1, n_output))
        self.activation_func = activation_func

    def forward(self, input_data):
        self.last_input = input_data
        linear_output = np.dot(input_data, self.weights) + self.biases
        self.last_output = self.activation_func(linear_output)
        return self.last_output

    def backward(self, d_output, learning_rate):
        d_linear_output = d_output * self.last_output * (1 - self.last_output)
        d_weights = np.dot(self.last_input.T, d_linear_output)
        d_biases = np.sum(d_linear_output, axis=0, keepdims=True)
        d_input = np.dot(d_linear_output, self.weights.T)
        self.weights += learning_rate * d_weights
        self.biases += learning_rate * d_biases
        return d_input


class SpamClassifier:
    def __init__(
            self,
            layers_config=((54, 'sigmoid'), (15, 'sigmoid'), (15, 'sigmoid'), (1, 'sigmoid')),
            learning_rate=0.01, epochs=2000, reg_lambda=0.01
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.layers = []

        for i in range(1, len(layers_config)):
            n_input, activation_func = layers_config[i - 1]
            n_output, _ = layers_config[i]
            layer = Layer(n_input, n_output, self.get_activation_func(activation_func))
            self.layers.append(layer)

    @staticmethod
    def get_activation_func(name):
        if name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'tanh':
            return lambda x: np.tanh(x)
        else:
            raise ValueError('Unknown activation function: ' + name)

    def score(self, data, features):
        y_pred = self.predict(data)
        return np.count_nonzero(y_pred == features) / features.shape[0]

    def train(self, train_data, features):
        n_samples, n_features = train_data.shape

        # Normalize data
        means = np.mean(train_data, axis=0)
        stds = np.std(train_data, axis=0)
        train_data = (train_data - means) / stds

        for i in range(self.epochs):
            # forward pass
            input_data = train_data
            for layer in self.layers:
                output = layer.forward(input_data)
                input_data = output

            # backward pass
            d_output = (features.reshape(-1, 1) - input_data) * input_data * (1 - input_data)
            d_input = d_output
            for layer in reversed(self.layers):
                d_input = layer.backward(d_input, self.learning_rate)

        # update weights with L2 regularization penalty
        for layer in self.layers:
            layer.weights += self.learning_rate * (-self.reg_lambda/n_samples * layer.weights)

    def predict(self, test_data):
        input_data = test_data
        for layer in self.layers:
            output = layer.forward(input_data)
            input_data = output
        y_pred = (input_data > 0.5).astype(int).flatten()
        return y_pred


