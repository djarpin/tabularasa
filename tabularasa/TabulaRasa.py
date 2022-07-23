import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tab_transformer_pytorch import TabTransformer
from tabularasa.MixedMonotonic import MixedMonotonicRegressor, MixedMonotonicNet
from tabularasa.SimultaneousQuantiles import SimultaneousQuantilesRegressor, SimultaneousQuantilesNet
from tabularasa.OrthonormalCertificates import OrthonormalCertificatesRegressor, OrthonormalCertificatesNet


class TabTransformerMixedMonotonicNet(MixedMonotonicNet):

    def forward(self, X_monotonic, X_categorical, X_non_monotonic, last_hidden_layer=False):
        h = self.non_monotonic_net(x_categ=X_categorical, x_cont=X_non_monotonic)
        return self.monotonic_net(X_monotonic, h, last_hidden_layer)


class TabTransformerSimultaneqousQuantilesNet(SimultaneousQuantilesNet):

    def forward(self, X_categorical, X_non_monotonic, qs, last_hidden_layer=False):
        h = self.non_monotonic_net(x_categ=X_categorical, x_cont=X_non_monotonic)
        return self.monotonic_net(qs, h, last_hidden_layer)


class TabulaRasaRegressor:

    def __init__(self,
                 df,
                 targets,
                 monotonic_constraints={},
                 **kwargs):
        # Set up data
        self.targets = targets
        self.monotonic_constraints = monotonic_constraints
        self.features = df.select_dtypes(include=['number', 'category', 'object']).columns.difference(targets).to_list()
        self._ingest(df)
        # Set up networks
        self._define_model()
        self._define_quantiles_model()
        self._define_uncertainty_model()

    def _define_model(self,
                      non_monotonic_net=None,
                      max_epochs=150,
                      lr=0.1,
                      optimizer=torch.optim.Adam,
                      layers=[128, 128, 32],
                      **kwargs):
        if non_monotonic_net is None:
            self.model_non_monotonic_net = TabTransformer(categories=tuple(self.categoricals_in),
                                                          num_continuous=len(self.numerics_non_monotonic),
                                                          dim=32,
                                                          dim_out=layers[-1],
                                                          depth=6,
                                                          heads=8,
                                                          attn_dropout=0.1,
                                                          ff_dropout=0.1,
                                                          mlp_hidden_mults=(4, 2),
                                                          mlp_act=torch.nn.ReLU())
        else:
            # Will this be saved?
            self.model_non_monotonic_net = non_monotonic_net
        self.model_layers = layers
        self.model = MixedMonotonicRegressor(TabTransformerMixedMonotonicNet,
                                             max_epochs=max_epochs,
                                             lr=lr,
                                             optimizer=optimizer,
                                             iterator_train__shuffle=True,
                                             module__non_monotonic_net=self.model_non_monotonic_net,
                                             module__dim_non_monotonic=len(self.numerics_non_monotonic) + sum(self.categoricals_out),
                                             module__dim_monotonic=len(self.monotonic_constraints),
                                             module__layers=layers,
                                             module__dim_out=len(self.targets),
                                             **kwargs)

    def _define_quantiles_model(self,
                                non_monotonic_net=None,
                                max_epochs=150,
                                lr=0.1,
                                optimizer=torch.optim.Adam,
                                layers=[128, 128, 32],
                                **kwargs):
        if non_monotonic_net is None:
            self.quantiles_model_non_monotonic_net = TabTransformer(categories=tuple(self.categoricals_in),
                                                                    num_continuous=len(self.numerics),
                                                                    dim=32,
                                                                    dim_out=layers[-1],
                                                                    depth=6,
                                                                    heads=8,
                                                                    attn_dropout=0.1,
                                                                    ff_dropout=0.1,
                                                                    mlp_hidden_mults=(4, 2),
                                                                    mlp_act=torch.nn.ReLU())
        else:
            self.quantiles_model_non_monotonic_net = non_monotonic_net
        self.quantiles_model = SimultaneousQuantilesRegressor(SimultaneousQuantilesNet,
                                                              max_epochs=max_epochs,
                                                              lr=lr,
                                                              optimizer=optimizer,
                                                              iterator_train__shuffle=True,
                                                              module__non_monotonic_net=self.quantiles_model_non_monotonic_net,
                                                              module__dim_non_monotonic=len(self.numerics_non_monotonic) + sum(self.categoricals_out),
                                                              module__layers=layers,
                                                              module__dim_out=len(self.targets),
                                                              **kwargs)

    def _define_uncertainty_model(self,
                                  dim_certificates=64,
                                  max_epochs=150,
                                  lr=0.1,
                                  optimizer=torch.optim.Adam,
                                  **kwargs):
        self.uncertainty_model = OrthonormalCertificatesRegressor(OrthonormalCertificatesNet,
                                                                  max_epochs=max_epochs,
                                                                  lr=lr,
                                                                  optimizer=optimizer,
                                                                  iterator_train__shuffle=True,
                                                                  module__dim_input=self.model_layers[-1] + len(self.monotonic_constraints),
                                                                  module__dim_certificates=dim_certificates)

    def _ingest(self, df):
        self._prepare_categoricals(df)
        self._prepare_numerics(df)

    def _prepare_categoricals(self, df):
        # TODO: allow for specifying dim in and out
        self.categoricals = df[self.features].select_dtypes(include=['category', 'object']).columns.to_list()
        self.categoricals_in = []
        self.categoricals_out = []
        self.categoricals_maps = []
        for c in self.categoricals:
            if df[c].dtype == 'category':
                u = sorted(df[c].cat.categories.values)
            else:
                u = sorted(df[c].unique())
            self.categoricals_in.append(len(u))
            self.categoricals_in.append(min(max(round(np.sqrt(len(u))), 1), 256))
            self.categoricals_maps.append({v: i for i, v in enumerate(u, 1)})

    def _prepare_numerics(self, df):
        numerics = df[self.features].select_dtypes(include='number').columns
        self.numerics = numerics.to_list()
        self.numerics_non_monotonic = numerics.difference(self.monotonic_constraints.keys()).to_list()
        df[self.numerics] = df[self.numerics].astype('float32')
        self.numerics_scaler = StandardScaler()
        self.numerics_scaler.fit(df[self.numerics])
        self.targets_scaler = StandardScaler()
        self.targets_scaler.fit(df[self.targets])

    def _preprocess(self, df):
        # What about overwriting?
        for c, m in zip(self.categoricals, self.categoricals_maps):
            df[c] = df[c].map(m).astype('int').fillna(0)
        df[self.numerics] = self.numerics_scaler.transform(df[self.numerics]).astype('float32')
        for c, s in self.monotonic_constraints.items():
            df[c] = df[c] * s
        return df

    def _preprocess_targets(self, df):
        df[self.targets] = self.targets_scaler.transform(df[self.targets])
        return df

    def _postprocess_targets(self, predictions):
        return self.targets_scaler.inverse_transform(predictions)

    def fit(self, df):
        # TODO: Should make this flexible in case there aren't categoricals or non monotonic continuous features
        df_processed = self._preprocess_targets(self._preprocess(df))
        X = {'X_monotonic': df_processed[sorted(self.monotonic_constraints.keys())].values,
             'X_categorical': df_processed[self.categoricals].values,
             'X_non_monotonic': df_processed[self.numerics_non_monotonic].values}
        y = df_processed[self.targets].values
        print('*** Training core model ***')
        self.model.fit(X, y)
        print('*** Training model to estimate uncertainty ***')
        h = self.model.predict(X, last_hidden_layer=True)
        self.uncertainty_model.fit(np.concatenate([X['X_monotonic'], h], axis=1))
        print('*** Training model to predict quantiles ***')
        X = {'X_categorical': df_processed[self.categoricals].values,
             'X_non_monotonic': df_processed[self.numerics].values}
        e = y - self.model.predict(X)
        self.quantiles_model.fit(X, e)

    def predict(self, df):
        df_processed = self._preprocess(df)
        X = {'X_monotonic': df_processed[sorted(self.monotonic_constraints.keys())].values,
             'X_categorical': df_processed[self.categoricals].values,
             'X_non_monotonic': df_processed[self.numerics_non_monotonic].values}
        y = self.model.predict(X)
        return self._postprocess_targets(y)

    def predict_quantile(self, df, q=None):
        df_processed = self._preprocess(df)
        y = self.model.predict({'X_monotonic': df_processed[sorted(self.monotonic_constraints.keys())].values,
                                'X_categorical': df_processed[self.categoricals].values,
                                'X_non_monotonic': df_processed[self.numerics_non_monotonic].values})
        e = self.quantiles_model.predict({'X_categorical': df_processed[self.categoricals].values,
                                          'X_non_monotonic': df_processed[self.numerics].values},
                                         q=q)
        return self._postprocess_targets(y + e)

    def estimate_uncertainty(self, df):
        df_processed = self._preprocess(df)
        X = {'X_monotonic': df_processed[sorted(self.monotonic_constraints.keys())].values,
             'X_categorical': df_processed[self.categoricals].values,
             'X_non_monotonic': df_processed[self.numerics_non_monotonic].values}
        h = self.model.predict(X, last_hidden_layer=True)
        return self.uncertainty_model.scaled_predict(np.concatenate([X['X_monotonic'], h], axis=1))
