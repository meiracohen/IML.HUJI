import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IMLearn.metrics.loss_functions import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_boost = AdaBoost(wl=DecisionStump, iterations=n_learners)
    ada_boost.fit(train_X, train_y)
    x = np.array(range(n_learners))
    y_test = []
    y_train = []
    for t in range(n_learners):
        loss_test = ada_boost.partial_loss(test_X, test_y, t+1)
        loss_train = ada_boost.partial_loss(train_X, train_y, t+1)
        y_test.append(loss_test)
        y_train.append(loss_train)
    fig = go.Figure()
    fig.add_scatter(x=x, y=y_test, name="test")
    fig.add_scatter(x=x, y=y_train, name="train")
    fig.update_layout(title_text="loss in function of number of learners")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    for t in T:
        y_symbol = np.where(test_y == -1, 0, 1)
        symbols = np.array(["circle", "x"])
        fig_t = go.Figure()
        fig_t.add_traces([decision_surface(lambda x: ada_boost.partial_predict(x, t),
                                           lims[0], lims[1], showscale=False),
                          go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
                                     marker=dict(color=test_y, symbol=symbols[y_symbol],
                                                 colorscale=[custom[0], custom[-1]],
                                                 line=dict(color="black", width=1)))])
        fig_t.update_layout(title_text="Dicision surface in " + str(t) + " learners")
        fig_t.show()

    # Question 3: Decision surface of best performing ensemble
    min_t = np.argmin(y_test) + 1
    y_symbol = np.where(test_y == -1, 0, 1)
    symbols = np.array(["circle", "x"])
    ac = accuracy(test_y, ada_boost.partial_predict(test_X, min_t))
    fig_min_t = go.Figure()
    fig_min_t.add_traces([decision_surface(lambda x: ada_boost.partial_predict(x, min_t),
                                       lims[0], lims[1], showscale=False),
                      go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=test_y, symbol=symbols[y_symbol],
                                             colorscale=[custom[0], custom[-1]],
                                             line=dict(color="black", width=1)))])
    fig_min_t.update_layout(title_text="Dicision surface of minimal loss size: " + str(min_t)
                            + " accuracy: " + str(ac))
    fig_min_t.show()

    # Question 4: Decision surface with weighted samples
    fig1 = go.Figure()
    sizes = ada_boost.D_ / max(ada_boost.D_) * 5
    y_symbol1 = np.where(train_y == -1, 0, 1)
    fig1.add_traces([decision_surface(ada_boost.predict,
                                       lims[0], lims[1], showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=train_y, symbol=symbols[y_symbol1], size=sizes,
                                             colorscale=[custom[0], custom[-1]],
                                             line=dict(color="black", width=1)))])
    fig1.update_layout(title_text="Decision surface with weighted samples")
    fig1.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
