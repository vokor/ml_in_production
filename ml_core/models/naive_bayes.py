import numpy as np
from sklearn.base import BaseEstimator
from scipy.special import logsumexp


class GaussianDistribution:
    def __init__(self, feature):
        self.mean = feature.mean(axis=0)
        self.std = feature.std(axis=0)

    def logpdf(self, value):
        return - 0.5 * np.log(2. * np.pi * self.std ** 2) - (value - self.mean) ** 2 / (2 * self.std ** 2)

    def pdf(self, value):
        return np.exp(self.logpdf(value))


class NaiveBayes(BaseEstimator):

    def fit(self, X, y, distributions=None):
        self.unique_labels = np.unique(y)

        # If distributions of features are not specified, they will be Gaussian
        if distributions is None:
            distributions = [GaussianDistribution] * X.shape[1]
        else:
            assert len(distributions) == X.shape[1]

        self.conditional_feature_distributions = {}
        for label in self.unique_labels:
            feature_distribution = []
            for column_index in range(X.shape[1]):
                feature_column = X[y == label, column_index]
                fitted_distr = distributions[column_index](feature_column)
                feature_distribution.append(fitted_distr)
            self.conditional_feature_distributions[label] = feature_distribution

        # Prior label distributions
        self.prior_label_distibution = {
            label: sum((y == label).astype(float)) / len(y)
            for label in self.unique_labels
        }

    def predict_log_proba(self, X):
        class_log_probas = np.zeros((X.shape[0], len(self.unique_labels)), dtype=float)

        for label_idx, label in enumerate(self.unique_labels):
            for idx in range(X.shape[1]):
                class_log_probas[:, label_idx] += self.conditional_feature_distributions[label][idx].logpdf(X[:, idx])
            class_log_probas[:, label_idx] += np.log(self.prior_label_distibution[label])

        for idx in range(X.shape[1]):
            class_log_probas -= logsumexp(class_log_probas, axis=1)[:, None]
        return class_log_probas

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        log_probas = self.predict_log_proba(X)
        return np.array([self.unique_labels[idx] for idx in log_probas.argmax(axis=1)])
