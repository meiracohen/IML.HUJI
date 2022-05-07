import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data_set = pd.read_csv(filename).dropna().drop_duplicates()
    features = data_set[["Year", "Month", "Day", "Country", "City"]]
    # # temp = data_set[["Temp"]]
    # # print(data_set["Country"].unique())
    # # features = pd.concat((features, pd.get_dummies(data_set["Country"])), axis=1)  #, drop_first=True
    # features = pd.concat((features, pd.get_dummies(data_set["City"])), axis=1)  #, drop_first=True
    date = pd.to_datetime(data_set["Date"]).dt.dayofyear
    features.insert(0, "DayOfYear", date, True)
    features.insert(features.shape[1], "Temp", data_set["Temp"], True)
    features = features.loc[features['Temp'] >= -50]
    features = features.loc[features['Temp'] <= 50]
    return features


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv",)

    # Question 2 - Exploring data for specific country
    data_israel = data.loc[data['Country'] == "Israel"]
    data_israel["Year"] = data_israel["Year"].astype(str)
    years = data_israel["Year"].unique()
    fig_il_temp = px.scatter(data_israel, x="DayOfYear", y="Temp", color="Year",
                             title="temperature as function of day of year")
    fig_il_temp.update_xaxes(title_text="day of year", title_standoff=25)
    fig_il_temp.update_yaxes(title_text="temperature", title_standoff=25)
    fig_il_temp.show()

    months = data_israel.groupby("Month").Temp.agg(np.std)
    fig_month_temp = px.bar(months, title="months temperatures std")
    fig_month_temp.update_yaxes(title_text="std temperature", title_standoff=25)
    fig_month_temp.show()

    # Question 3 - Exploring differences between countries
    country_month_temp_mean = data.groupby(["Country", "Month"]).Temp.agg(np.mean).reset_index()
    country_month_temp_std = data.groupby(["Country", "Month"]).Temp.agg(np.std).reset_index()
    fig_c_m_t = px.line(country_month_temp_mean, x="Month", y="Temp", color="Country",
                        error_y=country_month_temp_std["Temp"],
                        title="temerature of months in diffirent countries")
    fig_c_m_t.show()

    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y = split_train_test(data_israel["DayOfYear"], data_israel["Temp"])
    x_fig = np.linspace(1, 10, 10, dtype=int)
    y_fig = []
    for k in range(1, 11):
        polynomial = PolynomialFitting(k)
        polynomial.fit(train_x["DayOfYear"].to_numpy(), train_y["Temp"].to_numpy())
        loss = polynomial.loss(test_x["DayOfYear"].to_numpy(), test_y["Temp"].to_numpy())
        y_fig.append(round(loss, 2))
        print(round(loss, 2))
    fig = px.bar(x=x_fig, y=y_fig, title="loss as function of polynomial degree")
    fig.update_xaxes(title_text="degree", title_standoff=25)
    fig.update_yaxes(title_text="loss", title_standoff=25)
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    polynomial = PolynomialFitting(4)
    polynomial.fit(data_israel["DayOfYear"], data_israel["Temp"])
    countries = data["Country"].unique()
    countries = np.delete(countries, np.where(countries == "Israel"))
    loss_countries = []
    for country in countries:
        data_country = data.loc[data["Country"] == country]
        loss_countries.append(polynomial.loss(data_country["DayOfYear"].to_numpy(),
                                              data_country["Temp"].to_numpy()))
    fig5 = px.bar(x=countries, y=loss_countries, title="Israels' model error over the other countries")
    fig5.update_xaxes(title_text="country", title_standoff=25)
    fig5.update_yaxes(title_text="loss", title_standoff=25)
    fig5.show()
