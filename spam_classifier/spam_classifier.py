import numpy as np
from scipy.optimize import minimize


class SpamClassifier:
    def __init__(self, c=1):
        self.c = c
        self.weights = None
        self.bias = None

    def train(self, x, y):
        n, d = x.shape

        # minimize the objective function
        objective = lambda w: 0.5 * np.dot(w, w) + self.c * np.sum(np.maximum(0, 1 - y * (np.dot(x, w) - self.bias)))
        initial_w = np.zeros(d)
        self.weights = minimize(objective, initial_w, method='L-BFGS-B', bounds=[(-1, 1) for i in range(d)]).x

        # find support vectors and the bias term
        support_vectors = (1 - y * (np.dot(x, self.weights) - self.bias)) > 0
        self.bias = np.mean(y[support_vectors] - np.dot(x[support_vectors], self.weights))

    def predict(self, data):
        return np.sign(np.dot(data, self.weights) + self.bias)
