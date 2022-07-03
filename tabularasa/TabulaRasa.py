import pandas as pd
import torch


class TabulaRasaRegressor:

    def __init__(self, df, monotonic_constraints=None):
        self.monotonic_features = monotonic_features
        pass

    def ingest(self, df, monotonic_features=None):
        
        df.dtypes
        pass

    def fit(self, X, y):
        self.ingest(df)
        pass

    def predict(self, X):
        pass

    def predict_quantile(self, X, q=None):
        pass

    def predict_uncertainty(self, X)
