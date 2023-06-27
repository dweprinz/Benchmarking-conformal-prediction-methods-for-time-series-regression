from fortuna.conformal.regression.adaptive_conformal_regressor import (
    AdaptiveConformalRegressor,
)
from fortuna.conformal.regression.quantile import QuantileConformalRegressor
from fortuna.conformal.regression import enbpi
import numpy as np
import pandas as pd
from tqdm import tqdm
from sktime.utils.plotting import plot_series
from sktime.utils import plotting
import matplotlib.pyplot as plt


class CQR:
    def __init__(self):
        self.qcr = QuantileConformalRegressor()

    def create_conformal_interval(
        self,
        predictions: dict,
        y_test: np.ndarray,
        y_val: np.ndarray,
        T: int,
        tvt_split: list = [0.5, 0.25, 0.25],
    ):
        """Create conformal intervals from predictions and validation values

        Args:
            predictions (dict): Predicted values for each quantile
            y_test (np.ndarray): Y values of test data
            y_val (np.ndarray): Y values of validation data
            T (int): Amount of time series data samples
            tvt_split (list): Train val test split. Defaults to [0.5, 0.25, 0.25]

        Returns:
            np.ndarray: conformal intervals
        """
        assert np.sum(tvt_split) == 1, "You are not splitting the data correctly"

        # e.g. 500 : 750
        val_interval = range(int(tvt_split[0] * T), int(sum(tvt_split[0:2]) * T))

        # e.g. 750
        test_interval_start = int(sum(tvt_split[0:2]) * T)

        conformal_intervals = []

        # create interval for each test sample
        for j in tqdm(range(len(y_test))):
            # for first sample we don't concatenate data
            if j == 0:
                val_lower_bounds = predictions["lower"][0, val_interval, :]
                val_upper_bounds = predictions["upper"][0, val_interval, :]
                val_targets = y_val.reshape(-1, 1)
            else:
                # get the upper and lower intervals
                val_lower = predictions["lower"][0, val_interval, :]
                val_upper = predictions["upper"][0, val_interval, :]
                test_lower = predictions["lower"][
                    0, test_interval_start : test_interval_start + j, :
                ]
                test_upper = predictions["upper"][
                    0, test_interval_start : test_interval_start + j, :
                ]

                # concatenate test data until j-timestep
                val_lower_bounds = np.concatenate((val_lower, test_lower))
                val_upper_bounds = np.concatenate((val_upper, test_upper))
                val_targets = np.concatenate(
                    (y_val.reshape(-1, 1), np.array([y_test[:j]]).T)
                )

            # create conformal interval for current timestep j
            conformal_intervals.append(
                self.qcr.conformal_interval(
                    val_lower_bounds=val_lower_bounds,
                    val_upper_bounds=val_upper_bounds,
                    test_lower_bounds=predictions["lower"][
                        0, test_interval_start + j, :
                    ],
                    test_upper_bounds=predictions["upper"][
                        0, test_interval_start + j, :
                    ],
                    val_targets=val_targets,
                    error=0.1,
                )[0]
            )

        conformal_intervals = np.array(conformal_intervals)
        return conformal_intervals

    def plot_conformal_interval(
        self, X_all, y_series, predictions, X_test, conformal_intervals
    ):
        # plot time series data samples
        fig, ax = plot_series(y_series, markers=".")

        # plot the quantile lines
        for quantile, y_pred in predictions.items():
            ax.plot(X_all[:, 0], y_pred.reshape(-1, 1), label=f"Quantile: {quantile}")

        # fill the conformal interval
        ax.fill_between(
            X_test[:, 0].flatten(),
            *conformal_intervals.T,
            alpha=0.3,
            color="tab:purple",
            label="conformal interval",
        )

        # show legend
        ax.legend()

        ax.legend(loc="upper left")

        return ax


class ACI:
    def __init__(self):
        self.qcr = QuantileConformalRegressor()
        self.aci = AdaptiveConformalRegressor(conformal_regressor=self.qcr)

    def create_conformal_interval(
        self,
        predictions: dict,
        y_test: np.ndarray,
        y_val: np.ndarray,
        T: int,
        tvt_split: list = [0.5, 0.25, 0.25],
        gamma: float = 0.005,
    ):
        """Create conformal intervals from predictions and validation values

        Args:
            predictions (dict): Predicted values for each quantile
            y_test (np.ndarray): Y values of test data
            y_val (np.ndarray): Y values of validation data
            T (int): Amount of time series data samples
            tvt_split (list): Train val test split. Defaults to [0.5, 0.25, 0.25]

        Returns:
            _type_: _description_
        """
        assert np.sum(tvt_split) == 1, "Your not splitting the data correctly"

        # e.g. 500 : 750
        val_interval = range(int(tvt_split[0] * T), int(sum(tvt_split[0:2]) * T))

        # e.g. 750
        test_interval_start = int(sum(tvt_split[0:2]) * T)

        conformal_intervals = []

        errors = [0.1]  # initial is preferred alpha
        for j in range(len(y_test)):
            # for first sample we don't concatenate data
            if j == 0:
                val_lower_bounds = predictions["lower"][0, val_interval, :]
                val_upper_bounds = predictions["upper"][0, val_interval, :]
                val_targets = y_val.reshape(-1, 1)

            else:
                # get the upper and lower intervals
                val_lower = predictions["lower"][0, val_interval, :]
                val_upper = predictions["upper"][0, val_interval, :]
                test_lower = predictions["lower"][
                    0, test_interval_start : test_interval_start + j, :
                ]
                test_upper = predictions["upper"][
                    0, test_interval_start : test_interval_start + j, :
                ]

                # concatenate test data until j-timestep
                val_lower_bounds = np.concatenate((val_lower, test_lower))
                val_upper_bounds = np.concatenate((val_upper, test_upper))
                val_targets = np.concatenate(
                    (y_val.reshape(-1, 1), np.array([y_test[:j]]).T)
                )

            # create conformal interval for current timestep j
            conformal_intervals.append(
                self.aci.conformal_interval(
                    val_lower_bounds=val_lower_bounds,
                    val_upper_bounds=val_upper_bounds,
                    test_lower_bounds=predictions["lower"][
                        0, test_interval_start + j, :
                    ],
                    test_upper_bounds=predictions["upper"][
                        0, test_interval_start + j, :
                    ],
                    val_targets=val_targets,
                    error=errors[-1],
                )[0]
            )

            # update the error from current timestep

            error = self.aci.update_error(
                conformal_interval=conformal_intervals[-1],
                error=errors[-1],
                target=y_test[-1],
                target_error=0.1,
                gamma=gamma,
            )

            errors.append(error)

        conformal_intervals = np.array(conformal_intervals)
        return conformal_intervals

    def plot_conformal_interval(
        self, X_all, y_series, predictions, X_test, conformal_intervals
    ):
        # plot time series data samples
        fig, ax = plot_series(y_series)

        # plot the quantile lines
        for quantile, y_pred in predictions.items():
            ax.plot(X_all[:, 0], y_pred.reshape(-1, 1), label=f"Quantile: {quantile}")

        # fill the conformal interval
        ax.fill_between(
            X_test[:, 0].flatten(),
            *conformal_intervals.T,
            alpha=0.3,
            color="tab:purple",
            label="conformal interval",
        )

        # show legend
        ax.legend()

        ax.legend(loc="upper left")

        return ax


class DataFrameBootstrapper:
    def __init__(self, n_bs_samples: int):
        self.n_bs_samples = n_bs_samples

    def __call__(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        indices = np.random.choice(y.shape[0], size=(self.n_bs_samples, y.shape[0]))
        return indices, [(X[idx], y[idx]) for idx in indices]


class EnbPI:
    def __init__(self, n_bs_samples):
        self.enbpi = enbpi.EnbPI()
        self.bootstrapper = DataFrameBootstrapper(n_bs_samples)

    def bootstrap(self, X_train, y_train):
        bs_indices, bs_train_data = self.bootstrapper(X_train, y_train)
        return bs_indices, bs_train_data

    def train(self, model, n_bs_samples, bs_train_data, X_train, X_test):
        bs_train_preds = np.zeros((n_bs_samples, X_train.shape[0]))
        bs_test_preds = np.zeros((n_bs_samples, X_test.shape[0]))
        for i, batch in enumerate(bs_train_data):
            print(f"batch: {i}", end=" ")
            model.fit(*batch)

            # we concatenate so we the index is continuous
            preds = model.predict(np.concatenate([X_train, X_test])).flatten()
            bs_train_preds[i] = preds[: X_train.shape[0]]
            bs_test_preds[i] = preds[X_train.shape[0] :]

        return bs_train_preds, bs_test_preds

    def create_conformal_interval_online(
        self,
        bs_indices: np.ndarray,
        bs_train_preds: np.ndarray,
        bs_test_preds: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ):
        
        batch_size = 1
        conformal_intervals2 = np.zeros((len(y_test), 2))
        for i in range(0, len(y_test), batch_size):
            if i == 0:
                (
                    conformal_intervals2[:batch_size],
                    train_residuals,
                ) = self.enbpi.conformal_interval(
                    bootstrap_indices=bs_indices,
                    bootstrap_train_preds=bs_train_preds,
                    bootstrap_test_preds=bs_test_preds[:, :batch_size],
                    train_targets=y_train,
                    error=0.05,
                    return_residuals=True,
                )
            else:
                (
                    conformal_intervals2[i : i + batch_size],
                    train_residuals,
                ) = self.enbpi.conformal_interval_from_residuals(
                    train_residuals=train_residuals,
                    bootstrap_new_train_preds=bs_test_preds[:, i - batch_size : i],
                    bootstrap_new_test_preds=bs_test_preds[:, i : i + batch_size],
                    new_train_targets=y_test[i - batch_size : i],
                    error=0.05,
                )

        return conformal_intervals2

    def create_conformal_interval(
        self,
        bs_indices: np.ndarray,
        bs_train_preds: np.ndarray,
        bs_test_preds: np.ndarray,
        y_train: np.ndarray,
    ):
        conformal_intervals = self.enbpi.conformal_interval(
            bootstrap_indices=bs_indices,
            bootstrap_train_preds=bs_train_preds,
            bootstrap_test_preds=bs_test_preds,
            train_targets=y_train,
            error=0.1,
        )

        return conformal_intervals

    def plot_conformal_interval(
        self, X_all, y_series, predictions, X_test, conformal_intervals
    ):
        # plot time series data samples
        fig, ax = plot_series(y_series)

        ax.plot(X_all[:, 0], predictions, color="r")

        # fill the conformal interval
        ax.fill_between(
            X_test[:, 0].flatten(),
            *conformal_intervals.T,
            alpha=0.3,
            color="tab:purple",
            label="conformal interval",
        )

        # show legend
        ax.legend()

        ax.legend(loc="upper left")

        return ax


class EnCQR:
    def __init__(self, n_ensembles=3, alpha=0.1):
        # number of ensembles
        self.B = n_ensembles

        # confidence level
        self.alpha = alpha
        # quantiles to predict
        self.quantiles = [self.alpha / 2, 0.5, 1 - (self.alpha / 2)]

    def dataloader(self, df_train, df_val, df_test, differencing=False):
        if differencing:
            x_train, y_train = (
                df_train["y_diff_lag"].to_numpy().reshape(-1, 1),
                df_train["y_diff"].to_numpy(),
            )
            x_val, y_val = (
                df_val["y_diff_lag"].to_numpy().reshape(-1, 1),
                df_val["y_diff"].to_numpy(),
            )
            x_test, y_test = (
                df_test["y_diff_lag"].to_numpy().reshape(-1, 1),
                df_test["y_diff"].to_numpy(),
            )

            train_x, train_y = x_train, y_train

            # concat training and validation
            train_x = np.concatenate([train_x, x_val])
            train_y = np.concatenate([train_y, y_val])

            true_y = np.concatenate([df_train["y"].to_numpy(), df_val["y"].to_numpy()])

            # Make training batches
            batch_len = int(np.floor(train_x.shape[0] / self.B))
            train_data = []
            label = []
            for b in range(self.B):
                train_data.append(
                    [
                        train_x[b * batch_len : (b + 1) * batch_len],
                        train_y[b * batch_len : (b + 1) * batch_len],
                    ]
                )
                label.append(true_y[b * batch_len : (b + 1) * batch_len])

            return train_data, label, x_val, y_val, x_test, y_test

        x_train, y_train = df_train["y_lag"], df_train["y"]
        x_val, y_val = df_val["y_lag"], df_val["y"]
        x_test, y_test = df_test["y_lag"], df_test["y"]

        train_x, train_y = x_train.to_numpy(), y_train.to_numpy()

        # concat training and validation
        train_x = np.concatenate([train_x, x_val])
        train_y = np.concatenate([train_y, y_val])

        # Make training batches
        batch_len = int(np.floor(train_x.shape[0] / self.B))
        train_data = []
        label = []

        for b in range(self.B):
            train_data.append(
                [
                    train_x[b * batch_len : (b + 1) * batch_len],
                    train_y[b * batch_len : (b + 1) * batch_len],
                ]
            )
            label.append(train_y[b * batch_len : (b + 1) * batch_len])

        return (
            train_data,
            label,
            x_val.to_numpy(),
            y_val.to_numpy(),
            x_test.to_numpy(),
            y_test.to_numpy(),
        )

    def asym_nonconformity(self, label, low, high):
        """
        Compute the asymetric conformity score
        """
        error_high = label - high
        error_low = low - label
        return error_low, error_high

    def train(self, df_train, train_label, model, train_data):
        index = np.arange(self.B)

        # predictions made by each ensemble model
        self.ensemble_model_preds = []

        # time steps out
        time_steps_out = 1

        # dict containing LOO predictions
        dct_lo = {}
        dct_hi = {}
        for b in index:
            dct_lo[f"pred_{b}"] = []
            dct_hi[f"pred_{b}"] = []

        # training a model for each sub set Sb
        for b in index:
            print("Training model for Sb:", b, end=" ")
            f_hat_b = model
            # fit the model
            f_hat_b.fit(train_data[b][0].reshape(-1, 1), train_data[b][1])
            self.ensemble_model_preds.append(f_hat_b)

            # Leave-one-out predictions for each Sb
            indx_LOO = index[index != b]
            for idx in indx_LOO:
                # training predictions for each Sb
                pred = f_hat_b.predict(train_data[idx][0].reshape(-1, 1))

                # lower and upper bound predictions
                dct_lo[f"pred_{idx}"].append(pred[:, 0].reshape(-1, 1))
                dct_hi[f"pred_{idx}"].append(pred[:, 2].reshape(-1, 1))

        # aggregating the predictions
        f_hat_b_agg_low = np.zeros(
            (train_data[index[0]][0].shape[0], time_steps_out, self.B)
        )
        f_hat_b_agg_high = np.zeros(
            (train_data[index[0]][0].shape[0], time_steps_out, self.B)
        )

        for b in index:
            f_hat_b_agg_low[:, :, b] = np.mean(dct_lo[f"pred_{b}"], axis=0)
            f_hat_b_agg_high[:, :, b] = np.mean(dct_hi[f"pred_{b}"], axis=0)

        # compute residuals on the training data
        epsilon_low = []
        epsilon_hi = []
        for b in index:
            # print(train_label[b])
            e_low, e_high = self.asym_nonconformity(
                label=train_label[b][0],
                low=f_hat_b_agg_low[:, :, b],
                high=f_hat_b_agg_high[:, :, b],
            )
            epsilon_low.append(e_low)
            epsilon_hi.append(e_high)
        epsilon_low = np.array(epsilon_low).flatten()
        epsilon_hi = np.array(epsilon_hi).flatten()

        return epsilon_low, epsilon_hi

    def construct_PI(
        self, epsilon_low, epsilon_hi, test_x, test_y, test_label, test_start=None
    ):
        """Construct PIs for test data."""

        n_quantiles = 3
        time_steps_out = 1
        test_y = test_y.reshape(-1, 1)
        f_hat_t_batch = np.zeros(
            (test_y.shape[0], test_y.shape[1], n_quantiles, self.B)
        )

        # compute predictions for each model in the ensemble
        for b, model_b in enumerate(self.ensemble_model_preds):
            print("Creating test predictions for Sb", b)
            # we have for example 600 test samples and 10 models in the ensemble

            if not test_start:
                # output shape of prediction is (600, 1, 3) and we reshape it to (600, 3, 1)
                # f hat t batch has shape (600, 1, 3, 10)
                # print(test_x.shape)

                f_hat_t_batch[:, :, :, b] = model_b.predict(
                    test_x.reshape(-1, 1)
                ).reshape(-1, 1, n_quantiles)
            else:
                # print(test_x.shape)
                pred = model_b.predict(test_x.reshape(-1, 1), test_start).reshape(
                    -1, 1, n_quantiles
                )
                # print(pred.shape)
                # print("pred: ", pred)
                f_hat_t_batch[:, :, :, b] = pred

        # combine predictions from all models in the ensemble
        PI = np.mean(f_hat_t_batch, axis=-1)

        # initialize the conformal intervals
        conf_PI = np.zeros((test_y.shape[0], test_y.shape[1], n_quantiles))

        # insert the mean predictions
        conf_PI[:, :, 1] = PI[:, :, 1]

        # Conformalize prediction intervals on the test data
        for i in range(test_y.shape[0]):
            # get the lower and upper quantiles
            e_quantile_lo = np.quantile(epsilon_low, 1 - self.alpha / 2)
            e_quantile_hi = np.quantile(epsilon_hi, 1 - self.alpha / 2)

            # compute the conformal intervals
            conf_PI[i, :, 0] = PI[i, :, 0] - e_quantile_lo
            conf_PI[i, :, 2] = PI[i, :, 2] + e_quantile_hi

            # update epsilon with the last s steps
            e_lo, e_hi = self.asym_nonconformity(
                label=test_label[i], low=PI[i, :, 0], high=PI[i, :, 2]
            )
            epsilon_low = np.delete(epsilon_low, slice(0, time_steps_out, 1))
            epsilon_hi = np.delete(epsilon_hi, slice(0, time_steps_out, 1))
            epsilon_low = np.append(epsilon_low, e_lo)
            epsilon_hi = np.append(epsilon_hi, e_hi)

        return PI, conf_PI

    def plot_conformal_interval(
        self, X_all: np.ndarray, y_series: pd.Series, predictions: dict, X_test: np.ndarray, conformal_intervals: np.ndarray
    ):
        # plot time series data samples
        fig, ax = plot_series(y_series)

        # don't plot the pred line
        # ax.plot(X_all[:,0], predictions, color="r")

        # fill the conformal interval
        ax.fill_between(
            X_test[:, 0].flatten(),
            *conformal_intervals.T,
            alpha=0.3,
            color="tab:purple",
            label="conformal interval",
        )

        # show legend
        ax.legend()

        ax.legend(loc="upper left")

        return ax

    def plot_PIs(
        self,
        true,
        pred_mean,
        PI_low=None,
        PI_hi=None,
        conf_PI_low=None,
        conf_PI_hi=None,
        x_lims=None,
        scaler=None,
        title=None,
        label_pi=None,
    ):
        if scaler:
            true = scaler.inverse_transform_y(true)
            pred_mean = scaler.inverse_transform_y(pred_mean)
        true = true.flatten()
        pred_mean = pred_mean.flatten()

        plt.set_cmap("tab10")
        plt.cm.tab20(0)
        plt.figure(figsize=(12, 3.5))
        plt.plot(np.arange(true.shape[0]), true, label="True", color="k")
        plt.plot(pred_mean, label="Pred", color=plt.cm.tab10(1))

        if conf_PI_low is not None:
            if scaler:
                conf_PI_low = scaler.inverse_transform_y(conf_PI_low)
                conf_PI_hi = scaler.inverse_transform_y(conf_PI_hi)
                PI_low = scaler.inverse_transform_y(PI_low)
                PI_hi = scaler.inverse_transform_y(PI_hi)
            conf_PI_hi = conf_PI_hi.flatten()
            conf_PI_low = conf_PI_low.flatten()
            PI_hi = PI_hi.flatten()
            PI_low = PI_low.flatten()
            plt.fill_between(
                np.arange(true.shape[0]),
                conf_PI_low,
                conf_PI_hi,
                alpha=0.3,
                label="Conformalized",
            )
            plt.plot(
                PI_low, label="original", color=plt.cm.tab10(0), linestyle="dashed"
            )
            plt.plot(PI_hi, color=plt.cm.tab10(0), linestyle="dashed")

        if (conf_PI_low is None) and (PI_low is not None):
            if scaler:
                PI_low = scaler.inverse_transform_y(PI_low)
                PI_hi = scaler.inverse_transform_y(PI_hi)

            if label_pi is None:
                label_pi = "PI"
            PI_hi = PI_hi.flatten()
            PI_low = PI_low.flatten()
            plt.fill_between(
                np.arange(true.shape[0]), PI_low, PI_hi, alpha=0.3, label=label_pi
            )

        if x_lims is not None:
            plt.xlim(x_lims)
        plt.legend(loc="upper right")
        plt.grid()

        if title is not None:
            plt.title(title)

    plt.show()
