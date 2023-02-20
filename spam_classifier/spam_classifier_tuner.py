import random
import time

import matplotlib.pyplot as plt
from itertools import product

import numpy as np


class Tuner:
    def __init__(self, model, param_grid, max_iter=18000):
        self.results = None
        self.model = model
        self.param_grid = param_grid
        self.max_iter = max_iter

    def fit(self, train_samples, train_features, test_samples, test_features):
        param_combinations = list(product(*self.param_grid.values()))
        print(f"Total num of combinations: {len(param_combinations)}")

        results = {
            'params': [],
            'train_score': [],
            'test_score': [],
            'train_time': []
        }

        best_test_score = 0
        best_params = None
        iteration = 0

        random.shuffle(param_combinations)

        for params in param_combinations:
            iteration += 1

            results['params'].append(params)
            for param, value in zip(self.param_grid.keys(), params):
                setattr(self.model, param, value)

            start_time = time.time()
            self.model.train(train_samples, train_features)
            end_time = time.time()

            if end_time - start_time > 30:
                print(f"Skipping hyperparams={params} because training took longer than 30s")
                continue

            train_score = self.model.score(train_samples, train_features)
            test_score = self.model.score(test_samples, test_features)

            print(f"iteration={iteration}, hyperparams={params}, train_score={train_score}, test_score={test_score}, train_time={end_time-start_time:.2f}s")

            results['train_score'].append(train_score)
            results['test_score'].append(test_score)
            results['train_time'].append(end_time - start_time)

            if test_score > best_test_score:
                best_test_score = test_score
                best_params = params

            if iteration >= self.max_iter:
                break

        print(f"best_test_score={best_test_score}, best_params={best_params}")
        self.results = results

    def plot(self):
        plt.figure(figsize=(15, 10))

        # Plot learning curves
        epochs_values = [params[1] for params in self.results['params']]
        train_scores = self.results['train_score']
        test_scores = self.results['test_score']

        plt.subplot(2, 2, 1)
        plt.scatter(epochs_values, train_scores, label='Train', color='blue')
        plt.scatter(epochs_values, test_scores, label='Test', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot validation curves for learning rate and regularization lambda
        learning_rate_values = [params[2] for params in self.results['params']]
        train_scores = self.results['train_score']
        test_scores = self.results['test_score']

        plt.subplot(2, 2, 2)
        plt.scatter(learning_rate_values, train_scores, label='Train', color='blue')
        plt.scatter(learning_rate_values, test_scores, label='Test', color='red')
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Accuracy')
        plt.legend()

        reg_lambda_values = [params[3] for params in self.results['params']]
        train_scores = self.results['train_score']
        test_scores = self.results['test_score']

        plt.subplot(2, 2, 3)
        plt.scatter(reg_lambda_values, train_scores, label='Train', color='blue')
        plt.scatter(reg_lambda_values, test_scores, label='Test', color='red')
        plt.xscale('log')
        plt.xlabel('Regularization lambda')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot training time distribution
        training_times = self.results['train_time']
        training_params = [params for i, params in enumerate(self.results['params'])]
        training_times = [time for time in training_times]

        plt.subplot(2, 2, 4)
        plt.hist(training_times, bins=10)
        plt.xlabel('Training time (s)')
        plt.ylabel('Frequency')
        plt.xlim(0, 30)  # Excludes times greater than 30 seconds
        plt.title('Training Time Distribution (excluding >30s)')
        plt.ylim(0, len(training_params))

        plt.tight_layout()
        plt.savefig('results/hyperparams-vs-accuracy.png', dpi=300)
        plt.show()

        # Linear plot to compare training and test accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(train_scores, label='Train')
        plt.plot(test_scores, label='Test')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/accuracy-train-vs-test.png', dpi=300)
        plt.show()

