import numpy as np


class SpamClassifier:
    def __init__(self, k):
        self.k = k

    def train(self):
        pass

    def predict(self, data):
        return np.zeros(data.shape[0])
