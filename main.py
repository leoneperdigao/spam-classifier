import random

import numpy as np

from spam_classifier.spam_classifier import SpamClassifier
from spam_classifier.spam_classifier_tuner import Tuner

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
print("Shape of the spam training data set:", training_spam.shape)
print(training_spam)

testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
print("Shape of the spam testing data set:", testing_spam.shape)
print(testing_spam)

training_data = training_spam[:, 1:]
training_features = training_spam[:, 0].astype(np.float64)

testing_data = testing_spam[:, 1:]
testing_features = testing_spam[:, 0].astype(np.float64)

if __name__ == "__main__":
    model = SpamClassifier()

    param_grid = {
        'layers_config': (
            ((54, 'sigmoid'), (random.choice(np.arange(15, 50, 5)), 'sigmoid'), (random.choice(np.arange(15, 50, 5)), 'sigmoid'), (1, 'sigmoid')),
        ),
        'epochs': np.arange(1000, 4000, 1000),
        'learning_rate': np.arange(0.001, 1.001, 0.001),
        'reg_lambda': np.arange(0.01, 1.001, 0.01),
    }

    tuner = Tuner(model, param_grid)
    tuner.fit(training_data, training_features, testing_data, testing_features)
    tuner.plot()
