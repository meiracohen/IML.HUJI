from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        th = 0
        mis = np.Inf
        feature_index = 0
        sign = 1
        for j in range(X.shape[1]):
            new_th, new_mis = self._find_threshold(X[:, j], y, 1)
            if new_mis < mis:
                mis = new_mis
                th = new_th
                feature_index = j
                sign = 1
            new_th, new_mis = self._find_threshold(X[:, j], y, -1)
            if new_mis < mis:
                mis = new_mis
                th = new_th
                feature_index = j
                sign = -1
        self.threshold_ = th
        self.j_ = feature_index
        self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        # y_pred = np.zeros(X.shape[0])
        values = X[:, self.j_]
        y_pred = np.where(values < self.threshold_, -self.sign_, self.sign_)
        return y_pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        th = values[0]
        mis = np.inf
        for i in range(values.shape[0]):
            y_pred = np.where(values < values[i], -sign, sign)
            # new_mis = misclassification_error(labels, y_pred)
            new_mis = np.sum(np.where(np.sign(labels) != np.sign(y_pred), abs(labels), 0))
            if new_mis < mis:
                mis = new_mis
                th = values[i]
        return th, mis

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
        # return misclassification_error(y, y_pred)
        loss = np.sum(np.where(np.sign(y) != np.sign(y_pred), abs(y), 0))
        # loss = np.sum(np.sign(y) != np.sign(y_pred))
        # if normalize:
        return loss
        # return loss
