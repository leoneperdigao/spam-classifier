import random

import numpy as np

from spam_classifier.spam_classifier import SpamClassifier
from spam_classifier.spam_classifier_tuner import SpamClassifierTuner

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


def create_classifier():
    classifier = SpamClassifier()
    classifier.train(training_data, training_features)
    return classifier


if __name__ == "__main__":
    classifier = create_classifier()

    tuner = SpamClassifierTuner(
        hidden_layer_sizes=np.arange(1, 55, 1),
        learning_rates=np.arange(0.001, 1.001, 0.001),
        epochs_variants=np.arange(100, 4000, 100),
        reg_lambdas=np.arange(0.001, 10.001, 0.001),
    )

    param_distributions = {
        'hidden_layer_size': lambda: random.choice(tuner.hidden_layer_sizes),
        'learning_rate': lambda: random.choice(tuner.learning_rates),
        'epochs': lambda: random.choice(tuner.epochs_variants),
        'reg_lambda': lambda: random.choice(tuner.reg_lambdas),
    }
    tuner.random_search(
        training_data, training_features, testing_data, testing_features, param_distributions, n_iter=1000)