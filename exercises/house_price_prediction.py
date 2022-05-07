from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data_set = pd.read_csv(filename).dropna().drop_duplicates()
    features = data_set[["date", "bedrooms", "bathrooms", "sqft_living", "floors",
                         "waterfront", "view", "grade", "sqft_above",
                         "sqft_basement", "yr_renovated", "lat",
                         "sqft_living15", "sqft_lot15", "price"]]
    features.drop(features.loc[features["price"] <= 0].index, inplace=True)
    features["date"] = \
        pd.to_datetime(features["date"], infer_datetime_format=True).apply(lambda x: x.value)
    for feature in ["bedrooms", "bathrooms", "sqft_living", "floors",
                         "waterfront", "view", "yr_renovated", "price"]:
        features = features.loc[features[feature] >= 0]
    features.reset_index(drop=True, inplace=True)
    outcome = features["price"]
    del features["price"]
    return features, outcome


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        x_std = np.std(X[feature].values.tolist())
        y_std = np.std(y)
        cov = np.cov(X[feature].values.tolist(), y)
        corr = (cov / (x_std * y_std))[0][1]
        title = "correlation between " + feature + " and response: " + str(corr)
        fig = go.Figure()
        fig.add_scatter(x=X[feature].values.tolist(), y=y,
                        mode='markers', marker=dict(color="black"))
        fig.update_layout(title_text=title)
        fig.update_xaxes(title_text=feature, title_standoff=25)
        fig.update_yaxes(title_text="price", title_standoff=25)
        fig.write_image(output_path + '/' + feature + '.jpg')


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv",)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linear_regression = LinearRegression()
    x_fig = np.linspace(10, 100, 91, dtype=int)
    mean_loss = []
    std_loss = []
    for p in range(10, 101):
        loss_list = []
        for i in range(10):
            sub_train_x = train_x.sample(frac=p/100.0)
            sub_train_y = train_y.iloc[sub_train_x.index]
            linear_regression.fit(sub_train_x.to_numpy(), sub_train_y.to_numpy())
            curr_loss = linear_regression.loss(test_x.to_numpy(), test_y.to_numpy())
            loss_list.append(curr_loss)
        mean_loss.append(np.mean(loss_list))
        std_loss.append(np.std(loss_list))
    y_fig = mean_loss
    mean_loss = np.array(mean_loss)
    std_loss = np.array(std_loss)
    fig = go.Figure()
    fig.add_scatter(x=x_fig, y=y_fig, mode='markers', marker=dict(color="black"))
    fig.add_scatter(x=x_fig, y=mean_loss - 2 * std_loss, fill=None, mode="lines",
                    line=dict(color="lightgrey"), showlegend=False)
    fig.add_scatter(x=x_fig, y=mean_loss + 2 * std_loss, fill='tonexty',
                    mode="lines", line=dict(color="lightgrey"), showlegend=False)
    fig.update_layout(title_text="loss as function of the percentage of the training samples")
    fig.update_xaxes(title_text="percentage of sample", title_standoff=25)
    fig.update_yaxes(title_text="loss", title_standoff=25)
    fig.show()

