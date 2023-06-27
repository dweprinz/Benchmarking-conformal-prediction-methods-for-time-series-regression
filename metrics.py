import numpy as np
from sklearn.metrics import mean_squared_error


class Metrics:
    def __init__(self, y_test, y_pred, y_lower, y_upper):
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_lower = y_lower
        self.y_upper = y_upper

    def RMSE(self):
        return mean_squared_error(y_true=self.y_test, y_pred=self.y_pred, squared=False)

    def local_coverage(self, interval: tuple[int, int]):
        """Compute coverage for a specified interval."""
        l, r = interval
        not_covered = ~(
            (self.y_test[l:r] >= self.y_lower[l:r])
            & (self.y_test[l:r] <= self.y_upper[l:r])
        )

        return 1 - np.mean(not_covered)

    def PICP(self):
        in_the_range = np.sum(
            (self.y_test >= self.y_lower) & (self.y_test <= self.y_upper)
        )
        coverage = in_the_range / np.prod(self.y_test.shape)

        return coverage

    def PIAW(self):
        avg_length = np.mean(abs(self.y_upper - self.y_lower))
        return avg_length

    def PINAW(self):
        avg_length = np.mean(abs(self.y_upper - self.y_lower))
        R = self.y_test.max() - self.y_test.min()
        norm_avg_length = avg_length / R

        return norm_avg_length

    def CWC(self, eta: int = 30, alpha: float = 0.1):
        return (1 - self.PINAW()) * np.e ** (-eta * (self.PICP() - (1 - alpha)) ** 2)

    def computeAll(self, model_name="Unknown", eta=30, alpha=0.1):
        return {
            "model_name": model_name,
            "RMSE": self.RMSE(),
            "PICP": self.PICP(),
            "PIAW": self.PIAW(),
            "PINAW": self.PINAW(),
            "CWC": self.CWC(eta, alpha),
        }
