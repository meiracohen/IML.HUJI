from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X_data, y_data = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X_data, y_data))

        perceptron = Perceptron(callback=callback)
        perceptron.fit(X_data, y_data)


        # Plot figure
        x_fig = np.array(range(1, len(losses)+1))
        y_fig = losses
        fig = go.Figure()
        fig.add_scatter(x=x_fig, y=y_fig, mode='lines', marker=dict(color="black"))
        fig.update_layout(title_text=n + " loss as function of fit iteration")
        fig.update_xaxes(title_text="iterations", title_standoff=25)
        fig.update_yaxes(title_text="loss", title_standoff=25)
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")



def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X_data, y_data = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X_data, y_data)
        lda_predict = lda.predict(X_data)
        gaussian_naive_bayes = GaussianNaiveBayes()
        gaussian_naive_bayes.fit(X_data, y_data)
        gnb_predict = gaussian_naive_bayes.predict(X_data)



        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        x_fig = X_data[:, 0]
        y_fig = X_data[:, 1]

        accuracy_lda = accuracy(y_data, lda_predict)
        accuracy_gnb = accuracy(y_data, gnb_predict)
        title_lda = "LDA predict, accuracy is: " + str(round(accuracy_lda, 3))
        title_gnb = "gaussian naive bayes predict, accuracy is: " + str(round(accuracy_gnb, 3))

        fig = make_subplots(rows=1, cols=2, subplot_titles=[title_gnb, title_lda])
        fig.add_trace(
            go.Scatter(x=x_fig, y=y_fig, mode='markers',
                       marker=dict(color=gnb_predict, symbol=y_data)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_fig, y=y_fig, mode='markers',
                       marker=dict(color=lda_predict, symbol=y_data)),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=gaussian_naive_bayes.mu_[:, 0], y=gaussian_naive_bayes.mu_[:, 1],
                       mode="markers", marker=dict(color='black', symbol='x')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                       marker=dict(color='black', symbol='x')),
            row=1, col=2
        )

        for i in range(len(lda.classes_)):
            fig.add_trace(
                get_ellipse(lda.mu_[i], lda.cov_),
                row=1, col=2
            )
        for i in range(len(gaussian_naive_bayes.classes_)):
            fig.add_trace(
                get_ellipse(gaussian_naive_bayes.mu_[i], np.diag(gaussian_naive_bayes.vars_[i])),
                row=1, col=1
            )

        fig.update_layout(dict(title="Comparing between two classifiers"))
        fig.show()

        # Add traces for data-points setting symbols and colors
        raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
