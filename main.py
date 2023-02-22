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
    model = SpamClassifier(weights_file='./data/weights.npy')
    model.train(training_data, training_features, save_weights=False)
    print(model.score(testing_data, testing_features))  # 93%

    param_grid = {
        'layers_config': (
            ((54, 'sigmoid'), (5, 'sigmoid'), (5, 'sigmoid'), (1, 'sigmoid')),
            ((54, 'sigmoid'), (10, 'sigmoid'), (10, 'sigmoid'), (1, 'sigmoid')),
            ((54, 'sigmoid'), (15, 'sigmoid'), (15, 'sigmoid'), (1, 'sigmoid')),
            ((54, 'sigmoid'), (20, 'sigmoid'), (20, 'sigmoid'), (1, 'sigmoid')),
            ((54, 'sigmoid'), (25, 'sigmoid'), (25, 'sigmoid'), (1, 'sigmoid')),
            ((54, 'sigmoid'), (30, 'sigmoid'), (35, 'sigmoid'), (1, 'sigmoid')),
            ((54, 'sigmoid'), (40, 'sigmoid'), (40, 'sigmoid'), (1, 'sigmoid')),
            ((54, 'sigmoid'), (45, 'sigmoid'), (45, 'sigmoid'), (1, 'sigmoid')),
            ((54, 'sigmoid'), (50, 'sigmoid'), (50, 'sigmoid'), (1, 'sigmoid')),
        ),
        'epochs': np.arange(1000, 3000, 100),
        'learning_rate': np.logspace(-4, -1, 100),
        'reg_lambda': np.arange(0.01, 10.001, 0.01)
    }

    tuner = Tuner(model, param_grid, max_iter=1000)
    tuner.fit(training_data, training_features, testing_data, testing_features)
    tuner.plot()

