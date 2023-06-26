import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_series
from skforecast.ForecasterSarimax import ForecasterSarimax
from pmdarima import ARIMA as SARIMA


class ARIMA:
    def __init__(self) -> None:
        self.forecaster = ForecasterSarimax(
            regressor=SARIMA(order=(12, 1, 1), seasonal_order=(0, 0, 0, 0), maxiter=5),
        )

    def preprocess(self, data):
        # skforecast expects specific format
        data["datetime"] = pd.to_datetime(data["t"])
        data = data.set_index("datetime")
        data = data.asfreq("D")
        return data

    def sequential(self, Y_train: pd.Series, df_test, fh: list = [1], k: int = 5):
        # initialize training concatenation
        training = Y_train

        self.forecaster.fit(y=training)

        # we let arima forecast based on the training,
        assert k >= 1, "K should be a positive integer"
        predictions = pd.Series(dtype=object)

        print("Start forecasting")
        for i in tqdm(range(k)):
            # predict with predictive interval (not conformal)
            y_pred = self.forecaster.predict_interval(
                steps=1, alpha=0.1, interval=[5, 95]
            )

            # store predictions
            predictions = pd.concat([predictions, y_pred])

            # expand training data with recent y true values
            training = pd.concat([training, df_test.iloc[i : i + 1, :]["y"]])

            # fit forecaster with new observations
            self.forecaster.fit(y=training)

        return predictions

    def plot_interval(self, Y_series: pd.Series, predictions: pd.DataFrame):
        # plot the observations
        fig, ax = plot_series(Y_series)

        # plot the predictions
        predictions["pred"].plot(ax=ax, label="prediction", color="red")

        # plot the confidence intervals
        ax.fill_between(
            predictions.index,
            predictions["lower_bound"],
            predictions["upper_bound"],
            color="purple",
            alpha=0.3,
            label="90% interval",
        )
        ax.legend(loc="upper left")
