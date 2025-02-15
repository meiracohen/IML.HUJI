from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.pi_ = counts / len(y)
        self.mu_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        for i, c in enumerate(classes):
            self.mu_[i] = np.sum(X[np.where(y == c)], axis=0)
            self.mu_[i] /= counts[i]
            x_center = X[np.where(y == c)] - self.mu_[i]
            cov_ = x_center.T @ x_center
            self.vars_[i] = np.diag(cov_) / counts[i]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y_pred = np.zeros(X.shape[0])
        likelihood = self.likelihood(X)
        for j in range(X.shape[0]):
            x = likelihood[j]
            maxval = -np.inf
            argmax = self.classes_[0]
            for i, c in enumerate(self.classes_):
                if x[i] * self.pi_[i] > maxval:
                    maxval = x[i] * self.pi_[i]
                    argmax = c
            y_pred[j] = argmax
        return y_pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihood = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            cov = np.diag(self.vars_[i])

            two_pi_powered = (2 * np.pi) ** len(X[0])
            sqrt_value = 1 / (np.sqrt(two_pi_powered * np.linalg.det(cov)))
            X_centered = X - self.mu_[i].T
            pdf = [np.dot(x, np.dot(np.linalg.inv(cov), np.transpose(x))) for x in X_centered]
            pdf = [np.exp(-0.5 * p) * sqrt_value for p in pdf]

            likelihood[:,i] = pdf
        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)

