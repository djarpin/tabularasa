import pandas as pd
import numpy as np
import torch
import tabularasa.utils as utils


class TabulaRasaRegressor:

    def __init__(self, df, targets, monotonic_constraints=None):
        self.targets = targets
        self.monotonic_constraints = monotonic_constraints
        self.features = df.columns.difference(targets).to_list()
        self._ingest(df)

    def _ingest(self, df):
        self._prepare_categoricals(df)
        self._prepare_numerics(df)
        pass

    def _prepare_categoricals(self, df):
        self.categoricals = df.select_dtypes(include=['category', 'object']).columns.to_list()
        self.categoricals_in = []
        self.categoricals_out = []
        self.categoricals_maps = []
        for c in self.categoricals:
            u = df[c].unique().sort()
            self.categoricals_in.append(len(u))
            self.categoricals_in.append(min(max(round(np.sqrt(len(u))), 1), 256))
            self.categoricals_maps.append(dict(enumerate(u, 1)))

    def _prepare_numerics(self, df):
        # get standard scalers
        pass

    def _preprocess(self, X, y):
        # Encode
        # scale
        pass

    def fit(self, X, y):
        self._preprocess(df)
        pass

    def predict(self, X):
        self._preprocess(df)
        pass

    def predict_quantile(self, X, q=None):
        pass

    def predict_uncertainty(self, X):
        pass
