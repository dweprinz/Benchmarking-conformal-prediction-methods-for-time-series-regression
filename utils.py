import numpy as np
import statsmodels.api as sm
import pandas as pd
from sktime.utils.plotting import plot_series
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from conformal_methods import ACI, EnbPI
from metrics import Metrics


class SyntheticData:
    def __init__(self, n_samples, stationary=True):
        self.n_samples = n_samples
        self.stationary = stationary

    def generate_X(self):
        return np.random.uniform(0, 1, size=(self.n_samples, 6))

    def generate_epsilon(self, phi: list = [1, -1], theta: list = [1, 1]):
        return sm.tsa.arma_generate_sample(ar=phi, ma=theta, nsample=self.n_samples)

    def generate_Y(self):
        if self.stationary:
            Y = (
                10 * np.sin(np.pi * X[:, 0] * X[:, 1])
                + 20 * (X[:, 2] - 0.5) ** 2
                + 10 * X[:, 3]
                + 5 * X[:, 4]
                + 0 * X[:, 5]
            )
            return Y
        X = self.generate_X()

        epsilon = self.generate_epsilon()
        Y = (
            10 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3]
            + 5 * X[:, 4]
            + 0 * X[:, 5]
            + epsilon
        )

        return X, Y

    def Y_to_series(self, Y):
        return pd.Series(Y)

    def Y_to_df(self, X, Y):
        T = len(Y)
        df = pd.DataFrame({"t": np.arange(T), "y": Y})
        df_X = pd.DataFrame(X, columns=["X_1", "X_2", "X_3", "X_4", "X_5", "X_6"])
        df = df.join(df_X)
        return df, T

    def Y_to_csv(self, Y):
        name_to_store = f"synthetic_data/synthetic_{self.n_samples}_samples.csv"
        self.Y_to_series(Y).to_csv(name_to_store)

    def plot_Y_series(
        self, Y, x_label="Time $t$", y_label="$Y_t$", title="Synthetic Data"
    ):
        plot_series(self.Y_to_series(Y), x_label=x_label, y_label=y_label, title=title)


def create_subplots(
    X_all,
    X_test,
    df,
    predictions: list[tuple],
    titles: list,
    title="Quantile Regression",
):
    assert len(predictions) == len(
        titles
    ), "You should have the same amount of predictions as conformal method titles."
    # create subplot axes with a total size defined by figaspect
    fig, axes = plt.subplots(
        len(predictions), figsize=plt.figaspect(0.3 * len(predictions))
    )

    # general title for complete figure
    fig.suptitle(title)

    # for every subplot
    for i, ax in enumerate(axes):
        # get the predictions of current conformal method
        curr_predictions = predictions[i][0]

        # get the conformal intervals of current conformal method
        conformal_intervals = predictions[i][1]

        # create title equivalent to the conformal method
        ax.title.set_text(titles[i])

        # plot the time series data as a line
        ax.plot(X_all, df["y"])

        # plot the quantile lines
        for quantile, y_pred in curr_predictions.items():
            ax.plot(X_all, y_pred.reshape(-1, 1), label=f"Quantile: {quantile}")

        # fill the conformal interval
        ax.fill_between(
            X_test.flatten(),
            *conformal_intervals.T,
            alpha=0.3,
            color="C0",
            label="conformal interval",
        )

        # change figure labels
        ax.set_ylabel("$y$")
        ax.set_xlabel("$t$")

        # move the legend of the figure
        ax.legend()
        ax.legend(loc="upper left")

    # creates spacing between the subplots
    fig.tight_layout()


def load_file(parent, name, ext):
    """...

    Parameters
    ----------

    parent :
    name :
    ext :

    Returns
    -------
    file :
    """
    assert ext in ["pkl"], "ext must be pkl."
    path = parent + "/" + name + "." + ext
    if ext == "pkl":
        with open(path, "rb") as f:
            file = pickle.load(f)

    return file


def train_val_test(df, T, tvt_split=[0.6, 0.2, 0.2], shuffle=False):
    if shuffle:
        df_train_val = df[(df.index < int(sum(tvt_split[:2]) * T))]
        df_train_val = df_train_val.sample(frac=1).reset_index(drop=True)
        df_train = df_train_val[df_train_val.index < int(tvt_split[0] * T)]
        df_val = df_train_val[
            (int(tvt_split[0] * T) <= df_train_val.index)
            & (df_train_val.index < int(sum(tvt_split[:2]) * T))
        ]
        df_test = df[~(df.index < int(sum(tvt_split[:2]) * T))]

    else:
        df_train = df[df.index < int(tvt_split[0] * T)]
        df_val = df[
            (int(tvt_split[0] * T) <= df.index)
            & (df.index < int(sum(tvt_split[:2]) * T))
        ]
        df_test = df[~(df.index < int(sum(tvt_split[:2]) * T))]

    return df_train, df_val, df_test


def split_x_y(data, columns=["t", "y", "X_1", "X_2", "X_3", "X_4", "X_5", "X_6"]):
    x = data[columns[0]].to_numpy().reshape(-1, 1)
    y = data[columns[1]].to_numpy()
    return x, y

def compute_enbpi(model, df, df_train, df_val, df_test, tvt_split, T):
    X_bootstrap = np.concatenate([df_train["y_lag"].to_numpy().reshape(-1,1), 
                              df_val["y_lag"].to_numpy().reshape(-1,1)], axis=0)
    y_bootstrap = np.concatenate([df_train["y"].to_numpy(), df_val["y"].to_numpy()])

    n_bs_samples = 20
    enbpi = EnbPI(n_bs_samples=n_bs_samples)
    bs_indices, bs_train_data = enbpi.bootstrap(X_bootstrap, y_bootstrap)

    bs_train_preds, bs_test_preds = enbpi.train(model, n_bs_samples=n_bs_samples,
                                            bs_train_data=bs_train_data,
                                            X_train=X_bootstrap,
                                            X_test=df_test["y_lag"].to_numpy().reshape(-1,1))

    conformal_intervals = enbpi.create_conformal_interval_online(bs_indices=bs_indices,
                                bs_train_preds=bs_train_preds,
                                bs_test_preds=bs_test_preds,
                                y_train=y_bootstrap,
                                y_test=df_test["y"].to_numpy())

    preds = np.mean(np.concatenate([bs_train_preds, bs_test_preds], axis=1), axis=0)

    metric = Metrics(df_test["y"].to_numpy(), preds[df_test["t"]], conformal_intervals[:,0], conformal_intervals[:,1],)
    return metric.computeAll(model_name="EnbPI").values()
    


def compute_aci(model_predictions, df, df_train, df_val, df_test, tvt_split, T, gamma):
    aci = ACI()
    conformal_intervals = aci.create_conformal_interval(
        model_predictions,
        y_test=df_test["y"].to_numpy(),
        y_val=df_val["y"].to_numpy(),
        T=T,
        tvt_split=tvt_split,
        gamma=gamma,
    )

    metric = Metrics(
        df_test["y"].to_numpy(),
        model_predictions["middle"][0][df_test["t"], 0],
        conformal_intervals[:, 0],
        conformal_intervals[:, 1],
    )
    return metric.computeAll(model_name="ACI").values()


def optimize_conformal(
    param_range, model_predictions, df, df_train, df_val, df_test, tvt_split, T
):
    print(param_range)

    scores = []

    for param in tqdm(param_range):
        score = compute_aci(
            model_predictions,
            df,
            df_train,
            df_val,
            df_test,
            tvt_split,
            T=T,
            gamma=param,
        )
        scores.append(list(score))

    scores = np.array(scores)

    # find minimum PICP and return index
    minimum = np.argmin(scores[:, 2])

    # retrieve the corresponding gamma
    gamma = param_range[minimum]

    print("Example of grid searched results:")
    print("--------")
    print(scores[minimum])
    print("--------")

    return minimum, scores, gamma


def plot_PIs(
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
        plt.plot(PI_low, label="original", color=plt.cm.tab10(0), linestyle="dashed")
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


def plot_test_PIs(
    true,
    pred_mean,
    PI_low=None,
    PI_hi=None,
    conf_PI_low=None,
    conf_PI_hi=None,
    x_lims=None,
    scaler=None,
    label_pi=None,
    x_label=None,
    y_label=None,
    title=None,
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

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if title:
        plt.title(title)

    if conf_PI_low is not None:
        if scaler:
            conf_PI_low = scaler.inverse_transform_y(conf_PI_low)
            conf_PI_hi = scaler.inverse_transform_y(conf_PI_hi)
            PI_low = scaler.inverse_transform_y(PI_low)
            PI_hi = scaler.inverse_transform_y(PI_hi)
        conf_PI_hi = conf_PI_hi.flatten()
        conf_PI_low = conf_PI_low.flatten()
        plt.fill_between(
            np.arange(true.shape[0]),
            conf_PI_low,
            conf_PI_hi,
            alpha=0.3,
            label="Conformalized",
        )
        if PI_hi is not None and PI_low is not None:
            PI_hi = PI_hi.flatten()
            PI_low = PI_low.flatten()
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

        if PI_low is not None:
            PI_hi = PI_hi.flatten()
            PI_low = PI_low.flatten()
            plt.fill_between(
                np.arange(true.shape[0]), PI_low, PI_hi, alpha=0.3, label=label_pi
            )

    if x_lims is not None:
        plt.xlim(x_lims)
    plt.legend(loc="upper right")
    plt.grid()

    plt.show()
