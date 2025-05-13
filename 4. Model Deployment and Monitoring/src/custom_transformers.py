import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop, nonnegative_cols):
        self.cols_to_drop = cols_to_drop
        self.nonnegative_cols = nonnegative_cols

    def fit(self, X, y=None):
        return self

    # X is pd.DataFrame
    def transform(self, X):
        X_copy = X.copy()
        X_copy.drop(columns=self.cols_to_drop, errors="ignore", inplace=True)

        X_copy.replace(["?", "Error"], np.nan, inplace=True)
        X_copy["avg_frequency_login_days"] = X_copy["avg_frequency_login_days"].astype(
            float
        )

        for col in self.nonnegative_cols:
            X_copy.loc[X_copy[col] < 0, col] = np.nan

        return X_copy


# inspired by the Adapter design pattern ;)
class NaNImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, train_sample_size=30_000, verbose=True):
        self.train_sample_size = train_sample_size
        self.verbose = verbose
        self.imputer = NaNImputer(self.train_sample_size, self.verbose)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.imputer.impute(X)
