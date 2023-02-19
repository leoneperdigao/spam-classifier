from typing import List, Tuple

import numpy as np


class Layer:
    """
    A class representing a single layer in a neural network.

    Attributes:
        last_output (numpy.ndarray): The output of the layer during the last forward pass.
        last_input (numpy.ndarray): The input to the layer during the last forward pass.
        weights (numpy.ndarray): The weights of the layer.
        biases (numpy.ndarray): The biases of the layer.
        activation_func (function): The activation function of the layer.

    Methods:
        forward(input_data): Performs a forward pass through the layer.
        backward(d_output, learning_rate): Performs a backward pass through the layer and updates its weights and biases.
    """

    def __init__(self, n_input, n_output, activation_func):
        """
        Initializes a Layer object with the specified number of input and output nodes, and the specified activation function.

        Args:
            n_input (int): The number of input nodes.
            n_output (int): The number of output nodes.
            activation_func (function): The activation function of the layer.
        """
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
        """
        Performs a forward pass through the layer.

        Args:
            input_data (numpy.ndarray): The input data to the layer.

        Returns:
            numpy.ndarray: The output of the layer after applying the activation function to the linear combination of the input and the layer's weights and biases.
        """
        self.last_input = input_data
        linear_output = np.dot(input_data, self.weights) + self.biases
        self.last_output = self.activation_func(linear_output)
        return self.last_output

    def backward(self, d_output, learning_rate):
        """
        Performs a backward pass through the layer and updates its weights and biases.

        Args:
            d_output (numpy.ndarray): The gradient of the loss with respect to the output of the layer.
            learning_rate (float): The learning rate to use for updating the weights and biases.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of the layer.
        """
        d_linear_output = d_output * self.last_output * (1 - self.last_output)
        d_weights = np.dot(self.last_input.T, d_linear_output)
        d_biases = np.sum(d_linear_output, axis=0, keepdims=True)
        d_input = np.dot(d_linear_output, self.weights.T)
        self.weights += learning_rate * d_weights
        self.biases += learning_rate * d_biases
        return d_input


class SpamClassifier:
    """
    A class representing a spam classifier that uses a neural network with configurable architecture.

    Attributes:
        learning_rate (float): The learning rate to use during training.
        epochs (int): The number of training epochs.
        reg_lambda (float): The regularization parameter to use during training.
        layers (List[Layer]): A list of Layer objects representing the layers in the neural network.

    Methods:
        get_activation_func(name): Returns a function for the specified activation function name.
        score(data, features): Computes the accuracy of the classifier on the given data and labels.
        train(train_data, features): Trains the classifier on the given data and labels.
        predict(test_data): Predicts the labels for the given test data.
    """

    def __init__(
            self,
            layers_config: List[Tuple[int, str]] = ((54, 'sigmoid'), (15, 'sigmoid'), (15, 'sigmoid'), (1, 'sigmoid')),
            learning_rate: float = 0.01,
            epochs: int = 2000,
            reg_lambda: float = 0.01
    ):
        """
        Initializes a SpamClassifier object with the specified configuration.

        Args:
            layers_config (List[Tuple[int, str]]): A list of tuples specifying the number of nodes and activation function for each layer in the neural network.
            learning_rate (float): The learning rate to use during training.
            epochs (int): The number of training epochs.
            reg_lambda (float): The regularization parameter to use during training.
        """
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
    def get_activation_func(name: str):
        """
        Returns a function for the specified activation function name.

        Args:
            name (str): The name of the activation function.

        Returns:
            np.ufunc: The activation function.
        """
        if name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'tanh':
            return lambda x: np.tanh(x)
        else:
            raise ValueError('Unknown activation function: ' + name)

    def score(self, data: np.ndarray, features: np.ndarray) -> float:
        """
        Computes the accuracy of the classifier on the given data and labels.

        Args:
            data (numpy.ndarray): The data to evaluate the classifier on.
            features (numpy.ndarray): The labels corresponding to the data.

        Returns:
            float: The accuracy of the classifier on the given data and labels.
        """
        y_pred = self.predict(data)
        return np.count_nonzero(y_pred == features) / features.shape[0]

    def train(self, train_data: np.ndarray, features: np.ndarray) -> None:
        """
        Trains the classifier on the given data and labels.

        Args:
            train_data (numpy.ndarray): The data to train the classifier on.
            features (numpy.ndarray): The labels corresponding to the data.
        """
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

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the given test data.

        Args:
            test_data (numpy.ndarray): The data to predict the labels for.

        Returns:
            numpy.ndarray: The predicted labels.
        """
        input_data = test_data
        for i, layer in enumerate(self.layers):
            output = layer.forward(input_data)
            input_data = output

        # Convert output to binary labels (0 or 1) using appropriate threshold
        # based on the activation function used in the last layer
        last_activation_func = self.layers[-1].activation_func
        if last_activation_func == np.tanh or last_activation_func == np.maximum:
            y_pred = (input_data > 0).astype(int).flatten()
        else:
            y_pred = (input_data > 0.5).astype(int).flatten()

        return y_pred




