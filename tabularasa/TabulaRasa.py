import pandas as pd
import numpy as np
import torch
import tabularasa.utils as utils
from sklearn.preprocessing import StandardScaler


class TabulaRasaRegressor:

    def __init__(self, df, targets, monotonic_constraints=None):
        # Set up data
        self.targets = targets
        self.monotonic_constraints = monotonic_constraints
        self.features = df.select_dtypes(include=['number', 'category', 'object']).columns.difference(targets).to_list()
        self._ingest(df)
        # TODO: Set up networks
        self.regressor = MixedMonotonicRegressor(MixedMonotonicNet, )
        self.quantile_regressor = SimultaneousQuantileRegressor(SimultaneousQuantilesNet, )
        self.uncertainty_regressor = OrthonormalCertificatesRegressor(OrthonormalCertificatesNet, )

    def _ingest(self, df):
        self._prepare_categoricals(df)
        self._prepare_numerics(df)

    def _prepare_categoricals(self, df):
        self.categoricals = df[self.features].select_dtypes(include=['category', 'object']).columns.to_list()
        self.categoricals_in = []
        self.categoricals_out = []
        self.categoricals_maps = []
        for c in self.categoricals:
            u = df[c].unique().sort()
            self.categoricals_in.append(len(u))
            self.categoricals_in.append(min(max(round(np.sqrt(len(u))), 1), 256))
            self.categoricals_maps.append(dict(enumerate(u, 1)))

    def _prepare_numerics(self, df):
        self.numerics = df[self.features].select_dtypes(include='number').columns.to_list()
        self.numerics_scaler = StandardScaler()
        self.numerics_scaler.fit(df[self.numerics])
        self.targets_scaler = StandardScaler()
        self.targets_scaler.fit(df[self.targets])

    def _preprocess(self, df):
        # What about overwriting?
        for c, m in zip(self.categoricals, self.categoricals_maps):
            df[c] = df[c].map(m)
        df[self.numerics] = self.numerics_scaler.transform(df[self.numerics])
        for c, s in monotonic_constraints.items():
            df[c] = df[c] * s
        df[self.targets] = self.targets_scaler.transform(df[self.targets])
        return df

    def fit(self, df):
        df_processed = self._preprocess(df)
        # TODO: Should make this flexible in case there aren't categoricals
        pass

    def predict(self, df):
        self._preprocess(df)
        pass

    def predict_quantile(self, df, q=None):
        pass

    def predict_uncertainty(self, df):
        pass
