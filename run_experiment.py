# basemodels
from models import (
    QuantileLinearRegressor,
    QuantileForestRegressor,
    QuantileNeuralRegressor,
)
from arima import ARIMA
import quantnn

# metrics for comparison
from metrics import Metrics

# conformal methods to use
from conformal_methods import ACI, CQR, EnbPI, EnCQR

# data generation and subplots
from utils import (
    SyntheticData,
    create_subplots,
    train_val_test,
    split_x_y,
    compute_aci,
    optimize_conformal,
    plot_test_PIs,
)

# data science
import pandas as pd
import numpy as np
from sktime.utils.plotting import plot_series
from sktime.utils import plotting
from tqdm import tqdm
import matplotlib.pyplot as plt


def feature_engineering(df: pd.DataFrame):
    import warnings

    warnings.filterwarnings("ignore")

    # set column names correctly
    df = df[["t", "y"]]
    # create lagged y variable
    df["y_lag"] = df["y"].shift(1)
    # create differenced y variable
    df["y_diff"] = df["y"].diff()
    # create lagged differenced y variable
    df["y_diff_lag"] = df["y_diff"].shift(1)
    # drop na values due to lagging and differencing
    df = df.dropna().reset_index(drop=True)
    df.t = df.index
    T = len(df)
    # split the data
    tvt_split = [0.6, 0.2, 0.2]
    df_train, df_val, df_test = train_val_test(df, T, tvt_split=tvt_split)
    return df, df_train, df_val, df_test, T, tvt_split


def get_data(dataset: str):
    match dataset:
        case "synthetic":
            data = pd.read_csv("synthetic_data/synthetic_3000_samples.csv")
            data.columns = ["t", "y"]
        case "temperature":
            data = pd.read_csv("data/climate/DailyDelhiClimateTrain.csv")
            # get temperature data as Y values of series
            temperature_series = data["meantemp"]

            # get the date (datetime) as index of series
            temperature_series.index = data["date"].apply(pd.to_datetime)

            data = pd.DataFrame(temperature_series)

            # set column names correctly
            data["t"] = data.index
            data["y"] = data["meantemp"]
            data = data[["t", "y"]].reset_index(drop=True)
        case "power":
            data = pd.read_csv(
                "data/xu21_data/data/Wind_Hackberry_Generation_2019_2020.csv"
            )
            # aggregate df by day
            data["Date"] = pd.to_datetime(data["Date"].astype(str), format="%Y%m%d")
            data["Date"] = data["Date"].dt.date
            data = data.groupby("Date").sum().reset_index()

            data = data[["Date", "MWH"]]

            # get google stock closing as Y values of series
            power = data["MWH"]

            # get the date (datetime) as index of series
            power.index = data["Date"].apply(pd.to_datetime)

            data = pd.DataFrame(power)

            # set column names correctly
            data["t"] = data.index
            data["y"] = data["MWH"]
            data = data[["t", "y"]].reset_index(drop=True)
        case "google":
            data = pd.read_csv(
                "data/DIJA_stock/all_stocks_2006-01-01_to_2018-01-01.csv"
            )
            # specifically find google stock data
            data = data[data["Name"] == "GOOGL"]

            # get google stock closing as Y values of series
            stock = data["Close"]

            # get the date (datetime) as index of series
            stock.index = data["Date"].apply(pd.to_datetime)

            data = pd.DataFrame(stock)

            # set column names correctly
            data["t"] = data.index
            data["y"] = data["Close"]
            data = data[["t", "y"]].reset_index(drop=True)
        case "eurostox":
            data = pd.read_csv("data/eurostocks/yahoo_daily.csv")
            data.index = pd.to_datetime(data["Date"])
            data = pd.DataFrame(data["Close"])

            # set column names correctly
            data["t"] = data.index
            data["y"] = data["Close"]
            data = data[["t", "y"]].reset_index(drop=True)

        case _:
            print("Dataset not found")
            return None

    data = feature_engineering(data)
    return data


def experiment_arima(df: pd.DataFrame, T: int):
    import warnings

    warnings.filterwarnings("ignore")

    df_arima = df.copy()
    df_arima.t = pd.to_datetime(df_arima.t, unit="D")

    # split the data
    tvt_split = [0.6, 0.2, 0.2]
    df_arima_train, df_arima_val, df_arima_test = train_val_test(
        df_arima, T, tvt_split=tvt_split
    )
    arima = ARIMA()

    df_arima_train_val = pd.concat([df_arima_train, df_arima_val])
    df_arima_train_val = arima.preprocess(df_arima_train_val)
    df_arima_test = arima.preprocess(df_arima_test)

    # takes 10-30 min for 600 samples on CPU
    predictions = arima.sequential(
        df_arima_train_val.y, df_arima_test, k=len(df_arima_test)
    )

    # change prediction index so that it matches the test set
    predictions.index = df_arima_test.t

    # hacky code to reset the index
    df_arima_train, df_arima_val, df_arima_test = train_val_test(
        df_arima, T, tvt_split=tvt_split
    )

    df_arima = pd.DataFrame(
        {"model_name": [], "RMSE": [], "PICP": [], "PIAW": [], "PINAW": [], "CWC": []}
    )

    metric = Metrics(
        df_arima_test["y"].to_numpy(),
        predictions["pred"],
        predictions["lower_bound"].astype(float),
        predictions["upper_bound"].astype(float),
    )

    df_arima.loc[0] = metric.computeAll(model_name="ARIMA").values()

    return predictions, df_arima
