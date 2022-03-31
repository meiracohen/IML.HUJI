from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from utils import make_subplots
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univariate_gaussian = UnivariateGaussian()
    normal_samples = np.random.normal(10, 1, 1000)
    univariate_gaussian.fit(normal_samples)
    print('(', univariate_gaussian.mu_, ',', univariate_gaussian.var_, ')')

    # Question 2 - Empirically showing sample mean is consistent
    m = 100
    X1 = np.linspace(10, 1000, m, dtype=int)
    Y1 = [np.abs(univariate_gaussian.fit(normal_samples[:x]).mu_ - 10) for x in X1]
    fig1 = make_subplots(rows=1, cols=1) \
        .add_trace(go.Scatter(x=X1, y=Y1, mode='lines', marker=dict(color="black"), showlegend=False)) \
        .update_layout(title_text=r"$\text{Absolute distance between sample expectations and real one}$",
                       height=400)
    fig1.update_xaxes(title_text="samples amount", title_standoff=25)
    fig1.update_yaxes(title_text="distance", title_standoff=25)
    fig1.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    X2 = normal_samples
    Y2 = univariate_gaussian.pdf(normal_samples)
    fig2 = make_subplots(rows=1, cols=1) \
        .add_trace(go.Scatter(x=X2, y=Y2, mode='markers', marker=dict(color="red"), showlegend=False)) \
        .update_layout(title_text=r"$\text{PDF function on normal samples}$",
                       height=400)
    fig2.update_xaxes(title_text="normal samples", title_standoff=25)
    fig2.update_yaxes(title_text="pdf", title_standoff=25)
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multivariate_gaussian = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    normal_samples = np.random.multivariate_normal(mu, sigma, 1000)
    multivariate_gaussian.fit(normal_samples)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    m = 200
    f1 = np.linspace(-10, 10, m)
    f3 = np.linspace(-10, 10, m)
    log_likelihood = np.zeros((m, m))
    max_l_l = float('-inf'), None, None
    for i, f11 in enumerate(f1):
        for j, f33 in enumerate(f3):
            mu_graph = np.array([f11, 0, f33, 0])
            log_likelihood[i][j] = multivariate_gaussian.log_likelihood(mu_graph, sigma, normal_samples)
            if log_likelihood[i][j] > max_l_l[0]:
                max_l_l = log_likelihood[i][j], f11, f33
    heatmap = go.Figure()
    heatmap.add_heatmap(x=f1, y=f3, z=log_likelihood)
    heatmap.update_layout(dict(title="Log likelihood heatmap",))
    heatmap.update_xaxes(title_text="f1")
    heatmap.update_yaxes(title_text="f3")
    heatmap.show()

    # Question 6 - Maximum likelihood
    print("the max log likelihood is", max_l_l[0])
    print("f1 that gets it", round(max_l_l[1], 3), "f3 that gets it", round(max_l_l[2], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
