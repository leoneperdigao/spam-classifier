import time
import numpy as np
import matplotlib.pyplot as plt

from .spam_classifier import SpamClassifier


class SpamClassifierTuner(SpamClassifier):
    def __init__(self, hidden_layer_sizes, learning_rates, epochs_variants, reg_lambdas):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rates = learning_rates
        self.epochs_variants = epochs_variants
        self.reg_lambdas = reg_lambdas

    def score(self, data, features):
        y_pred = self.predict(data)
        return np.count_nonzero(y_pred == features) / features.shape[0]

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)

    def random_search(self, X_train, y_train, X_test, y_test, param_distributions, n_iter=10):
        best_score = 0
        best_params = None
        results = []

        for i in range(n_iter):
            params_dict = {param: distribution() for param, distribution in param_distributions.items()}
            self.set_params(**params_dict)
            start_time = time.time()
            self.train(X_train, y_train)
            train_time = time.time() - start_time
            train_score = self.score(X_train, y_train)
            test_score = self.score(X_test, y_test)
            print(f"Iteration {i+1}: Train Score = {train_score:.3f}, Test Score = {test_score:.3f}, Train Time = {train_time:.3f} sec")
            print(f"Parameters {params_dict}")
            if test_score > best_score:
                best_score = test_score
                best_params = params_dict

            # Add results to list
            result = {
                'params': params_dict,
                'train_score': train_score,
                'test_score': test_score,
                'train_time': train_time
            }
            results.append(result)

        print(f"\nBest Parameters: {best_params}")
        print(f"Best Test Score: {best_score:.3f}")

        SpamClassifierTuner.plot_results(param_distributions, results)

    @staticmethod
    def plot_results(param_distributions, results):
        fig, ax = plt.subplots()
        ax.plot([result['train_score'] for result in results], label="Train Score")
        ax.plot([result['test_score'] for result in results], label="Test Score")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Performance by Iteration")
        ax.legend()
        plt.show()

        # Create scatter plot
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.ravel()
        for i, param in enumerate(param_distributions.keys()):
            param_values = [result['params'][param] for result in results]
            test_scores = [result['test_score'] for result in results]
            axs[i].scatter(param_values, test_scores)
            axs[i].set_xlabel(param)
            axs[i].set_ylabel('Test Score')
        plt.show()