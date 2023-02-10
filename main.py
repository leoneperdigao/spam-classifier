import numpy as np

from spam_classifier.spam_classifier import SpamClassifier


training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(np.int)
print("Shape of the spam training data set:", training_spam.shape)
print(training_spam)

testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(np.int)
print("Shape of the spam testing data set:", testing_spam.shape)
print(testing_spam)


def create_classifier():
    classifier = SpamClassifier(k=1)
    classifier.train()
    return classifier


classifier = create_classifier()