import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sktime.utils.plotting import plot_series
from sktime.utils import plotting
from tqdm import tqdm

from quantile_forest import RandomForestQuantileRegressor
import quantnn
from sklearn.linear_model import QuantileRegressor

# https://pypi.org/project/quantnn/
# !pip install quantnn

# https://pypi.org/project/quantile-forest/
# !pip install quantile-forest

# https://neuralprophet.com/contents.html
# !pip install neuralprophet


class QuantileLinearRegressor:
    def __init__(self, quantiles, alpha=1):
        self.quantiles = quantiles

        self.QRs = [
            QuantileRegressor(quantile=q, alpha=alpha, solver="highs")
            for q in quantiles
        ]

    def predict_quantiles(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_all: np.ndarray,
        quantiles: dict = {"lower": 0.05, "upper": 0.95},
    ):
        """Fit conditional quantiles and make predictions for all samples.

        Args:
            X_train (np.ndarray): Training data for quantile fit
            y_train (np.ndarray): Training labels for quantile fit
            X_all (np.ndarray): All X data from time series data
            quantiles (dict, optional): Quantiles to fit (bound, quantile). Defaults to {"lower": 0.05, "upper" : 0.95}.

        Returns:
            dict: y_preds on all X data for each quantile
        """
        predictions = {}

        # fit each prompted quantile and predict on all samples
        for bound, quantile in quantiles.items():
            self.fit(X_train, y_train)
            y_pred = self.predict(X_all)

            predictions[str(bound)] = y_pred

        return predictions

    def fit(self, X_train, y_train):
        # fit all quantile regressors
        for QR in self.QRs:
            QR.fit(X_train, y_train)

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], len(self.quantiles)))
        for i, QR in enumerate(self.QRs):
            y_pred[:, i] = QR.predict(X)
        return y_pred

    def plot_quantiles(
        self, X_all: np.ndarray, y_series: pd.Series, predictions: dict[str, np.ndarray]
    ):
        # plot the time series samples
        fig, ax = plot_series(y_series)

        # plot the quantile lines
        for quantile, y_pred in predictions.items():
            ax.plot(X_all[:, 0], y_pred.reshape(-1, 1), label=f"Quantile: {quantile}")

        ax.legend()


class QuantileForestRegressor:
    def __init__(self, y_real, quantiles=[0.05, 0.95], differencing=False, **kwargs):
        self.qrf = RandomForestQuantileRegressor(**kwargs)
        self.y_real = y_real
        self.quantiles = quantiles

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.qrf.fit(X_train, y_train)

    def predict(self, X_all: np.ndarray, start: int = 0):
        # we get X_all but predict sequentially for each sample

        preds = np.zeros((X_all.shape[0], len(self.quantiles)))

        # for every time step
        for i, value in tqdm(enumerate(X_all[:, 0])):
            # get the lagged y_diff and predict the two quantiles of current y_diff
            pred_diff = self.qrf.predict(value.reshape(1, -1), quantiles=self.quantiles)

            if i == 0:
                preds[i] = pred_diff[0] + self.y_real[start]
            else:
                preds[i] = pred_diff[0] + self.y_real[start + i - 1]
        return preds

    def plot_quantiles(
        self, X_all: np.ndarray, y_series: pd.Series, predictions: dict[str, np.ndarray]
    ):
        # plot the time series samples
        fig, ax = plot_series(y_series)

        # plot the quantile lines
        for quantile, y_pred in predictions.items():
            ax.plot(X_all[:, 0], y_pred.reshape(-1, 1), label=f"Quantile: {quantile}")

        ax.legend()


class QuantileNeuralRegressor:
    def __init__(
        self,
        quantiles=[0.05, 0.95],
        n_inputs=1,
        model_params: tuple[int, int, str] = (4, 256, "relu"),
    ):
        self.quantiles = quantiles
        self.n_inputs = n_inputs
        self.model_params = model_params
        self.qrnn = quantnn.QRNN(
            quantiles=self.quantiles, n_inputs=self.n_inputs, model=self.model_params
        )

    def fit(self, X_train, y_train, n_epochs: int = 15):
        training_data = (X_train, y_train)
        logs = self.qrnn.train(training_data=training_data, n_epochs=n_epochs)
        return logs

    def predict(self, lags: np.ndarray):
        # Perform prediction with the trained models
        y_pred = self.qrnn.predict(lags)

        return y_pred.numpy()

    def plot_quantiles(self, X_all, y_series, predictions):
        # plot the time series samples
        fig, ax = plot_series(y_series)

        # plot the quantile lines
        for quantile, y_pred in predictions.items():
            ax.plot(X_all[:, 0], y_pred.reshape(-1, 1), label=f"Quantile: {quantile}")

        ax.legend()


class QuantileNeuralProphetRegressor:
    def __init__(self, quantiles=[0.05, 0.95], **kwargs):
        self.quantiles = quantiles
        self.NN = NeuralProphet(quantiles=self.quantiles, **kwargs)

    def preprocess_data(self, df: pd.DataFrame):
        # t and Y columns are expected to be named ds and y
        data = df.copy()
        data.columns = ["ds", "y"]
        data["ds"] = data["ds"].apply(pd.to_datetime)
        return data

    def fit(self, df_train: pd.DataFrame):
        # Fit the model with the observations

        # when transforming integers to datetime, the unit is assumed to be nanoseconds
        self.NN.fit(df_train, freq="ns")

    def predict(
        self,
        data: pd.DataFrame,
        columns: list = ["t", "y", "yhat1", "lower", "upper", "ar1", "trend"],
    ):
        # Perform prediction with the trained models
        y_pred = self.NN.predict(df=data)
        display(y_pred)
        y_pred.columns = columns
        return y_pred

    def plot_quantiles(self, X_all, y_series, predictions):
        # plot the time series samples
        fig, ax = plot_series(y_series)

        # plot the quantile lines
        for quantile, y_pred in predictions.items():
            ax.plot(X_all[:, 0], y_pred.reshape(-1, 1), label=f"Quantile: {quantile}")

        ax.legend()
