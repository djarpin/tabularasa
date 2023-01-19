import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from skorch.callbacks import EarlyStopping
from tab_transformer_pytorch import TabTransformer
from tabularasa.gumnn.MultidimensionnalMonotonicNN import SlowDMonotonicNN
from tabularasa.MixedMonotonic import MixedMonotonicRegressor, MixedMonotonicNet
from tabularasa.SimultaneousQuantiles import SimultaneousQuantilesRegressor
from tabularasa.OrthonormalCertificates import OrthonormalCertificatesRegressor, OrthonormalCertificatesNet


#####################
# Network definitions
#####################


class TabTransformerMixedMonotonicNet(MixedMonotonicNet):

    def forward(self, X_monotonic, X_categorical, X_non_monotonic, last_hidden_layer=False):
        '''
        Update `MixedMonotonicNet().forward()` to support separate categorical and non-monotonic continuous inputs

        Parameters
        ----------
        X_monotonic : torch.Tensor
            Monotonically constrained feature values
        X_categorical : torch.Tensor
            Cateogorical feature label-encoded values
        X_non_monotonic : torch.Tensor
            Non-monotonically constrained continuous feature values
        last_hidden_layer : bool, optional
            Return activations from last hidden layer in the neural network
            Needed for Orthonormal Certificates to estimate epistemic uncertainty (defaults to False)

        Returns
        -------
        torch.Tensor
            Output from monotonically constrained neural network
        '''
        h = self.non_monotonic_net(x_categ=X_categorical, x_cont=X_non_monotonic)
        return self.monotonic_net(X_monotonic, h, last_hidden_layer)


class TabTransformerSimultaneousQuantilesNet(torch.nn.Module):

    def __init__(self,
                 non_monotonic_net,
                 dim_non_monotonic,
                 dim_monotonic,
                 layers=[512, 512, 64],
                 dim_out=1,
                 integration_steps=50,
                 device='cpu'):
        '''
        Create alternate SimultaneousQuantilesNet to support monotonically constrained features

        Parameters
        ----------
        non_monotonic_net : torch.nn.Module
            The initialized PyTorch network for non-monotonically constrained features
            The `.forward()` method for this network must accept a single argument: X_non_monotonic
        dim_non_monotonic : int
            Output dimension of `non_monotonic_net`
        layers : list[int], optional
            Neurons in each hidden layer (defaults to [512, 512, 64])
        dim_out : int, optional
            Output dimension of network (number of target variables to predict) (defaults to 1)
        integration_steps : int, optional
            Number of integration steps in Clenshaw-Curtis Quadrature Method (defaults to 50)
        device : str, optional
            'cpu' or 'cuda:0' (defaults to 'cpu')

        Returns
        -------
        None
            Initializes Simultaneous Quantiles neural network
        '''
        super().__init__()
        self.non_monotonic_net = non_monotonic_net
        self.monotonic_net = SlowDMonotonicNN(dim_monotonic,
                                              dim_non_monotonic + 1,
                                              layers,
                                              dim_out,
                                              integration_steps,
                                              device)

    def forward(self, X_monotonic, X_categorical, X_non_monotonic, qs, last_hidden_layer=False):
        '''
        Create alternate `SimultaneousQuantilesNet().forward()` to support separate categorical and non-monotonic continuous inputs

        Parameters
        ----------
        X_categorical : torch.Tensor
            Cateogorical feature label-encoded values
        X_non_monotonic : torch.Tensor
            Non-monotonically constrained continuous feature values
        qs : torch.Tensor
            Quantile for each record
        last_hidden_layer : bool, optional
            Return activations from last hidden layer in the neural network
            Needed for Orthonormal Certificates to estimate epistemic uncertainty (defaults to False)

        Returns
        -------
        torch.Tensor
            Output from Simultaneous Quantiles neural network
        '''
        h = self.non_monotonic_net(x_categ=X_categorical, x_cont=X_non_monotonic)
        # TODO: Figure out if there is a way to keep qs monotonic
        return self.monotonic_net(X_monotonic, torch.cat([h, qs], 1), last_hidden_layer)


##################
# Regressor object
##################


class TabulaRasaRegressor:

    def __init__(self,
                 monotonic_constraints={},
                 **kwargs):
        '''
        Initialize regressor with monotonically constrained feature support
        As well as epistemic and aleatoric uncertainty estimates

        Paramters
        ---------
        monotonic_constraints : dict
            Lookup where keys are column names of features to be monotonically constrained.
            And values are 1 or -1 to dictate increasing or decreasing relationship with the target (respectively)
        
        Returns
        -------
        None
            Initializes model object
        '''
        self.trained = False
        self.monotonic_constraints = monotonic_constraints
        self.kwargs = kwargs

    def _setup(self, X, y):
        # Set up data
        self.features = X.select_dtypes(include=['number', 'category', 'object']).columns.to_list()
        self.dim_y = y.shape[1]
        self._ingest(X, y)
        # Set up networks
        # TODO: Get all kwargs working deeper into the model definitions
        self._define_model(**self.kwargs)
        self._define_quantiles_model(**self.kwargs)
        self._define_uncertainty_model(**self.kwargs)

    def _define_model(self,
                      non_monotonic_net=None,
                      max_epochs=100,
                      lr=0.01,
                      optimizer=torch.optim.Adam,
                      layers=[128, 128, 32],
                      **kwargs):
        '''
        Sets up base model for montonically constrained predictions

        Parameters
        ----------
        non_monotonic_net : torch.nn.Module, optional
            The initialized PyTorch network for non-monotonically constrained features
            The `.forward()` method for this network must accept a two arguements: x_categ and x_cont
            (defaults to TabTransformer)
        max_epochs : int, optional
            Number of epochs to train for (defaults to 100)
        lr : float, optional
            Learning rate for optimization (defaults to 0.01)
        optimizer : torch.optim.Optimizer
            PyTorch optimizer to use in training (defaults to torch.optim.Adam)
        layers : list[int], optional
            Neurons in each hidden layer (defaults to [128, 128, 32])

        Returns
        -------
        None
        '''
        if non_monotonic_net is None:
            self.model_non_monotonic_net = TabTransformer(categories=tuple(self.categoricals_in),
                                                          num_continuous=len(self.numerics_non_monotonic),
                                                          dim=32,
                                                          dim_out=len(self.numerics_non_monotonic) + sum(self.categoricals_out),
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
        # TODO: fix this so that if non_monotonic_net is specified we don't use TabTransformerMixedMonotonicNet
        self.model = MixedMonotonicRegressor(TabTransformerMixedMonotonicNet,
                                             max_epochs=max_epochs,
                                             lr=lr,
                                             optimizer=optimizer,
                                             iterator_train__shuffle=True,
                                             callbacks=[EarlyStopping(patience=max_epochs // 10)],
                                             module__non_monotonic_net=self.model_non_monotonic_net,
                                             module__dim_non_monotonic=len(self.numerics_non_monotonic) + sum(self.categoricals_out),
                                             module__dim_monotonic=len(self.monotonic_constraints),
                                             module__layers=layers,
                                             module__dim_out=self.dim_y,
                                             **kwargs)

    def _define_quantiles_model(self,
                                non_monotonic_net=None,
                                max_epochs=100,
                                lr=0.01,
                                optimizer=torch.optim.Adam,
                                layers=[128, 128, 32],
                                **kwargs):
        '''
        Sets up Simultaneous Quantiles model to estimate aleatoric uncertainty around base model

        Parameters
        ----------
        non_monotonic_net : torch.nn.Module, optional
            The initialized PyTorch network for non-monotonically constrained features
            The `.forward()` method for this network must accept a two arguements: x_categ and x_cont
            (defaults to TabTransformer)
        max_epochs : int, optional
            Number of epochs to train for (defaults to 100)
        lr : float, optional
            Learning rate for optimization (defaults to 0.01)
        optimizer : torch.optim.Optimizer
            PyTorch optimizer to use in training (defaults to torch.optim.Adam)
        layers : list[int], optional
            Neurons in each hidden layer (defaults to [128, 128, 32])

        Returns
        -------
        None
        '''
        if non_monotonic_net is None:
            self.quantiles_model_non_monotonic_net = TabTransformer(categories=tuple(self.categoricals_in),
                                                                    num_continuous=len(self.numerics_non_monotonic),
                                                                    dim=32,
                                                                    dim_out=len(self.numerics_non_monotonic) + sum(self.categoricals_out),
                                                                    depth=6,
                                                                    heads=8,
                                                                    attn_dropout=0.1,
                                                                    ff_dropout=0.1,
                                                                    mlp_hidden_mults=(4, 2),
                                                                    mlp_act=torch.nn.ReLU())
        else:
            self.quantiles_model_non_monotonic_net = non_monotonic_net
        # TODO: fix this so that if non_monotonic_net is specified we don't use TabTransformerSimultaneousQuantilesNet
        self.quantiles_model = SimultaneousQuantilesRegressor(TabTransformerSimultaneousQuantilesNet,
                                                              max_epochs=max_epochs,
                                                              lr=lr,
                                                              optimizer=optimizer,
                                                              callbacks=[EarlyStopping(patience=max_epochs // 10)],
                                                              iterator_train__shuffle=True,
                                                              module__non_monotonic_net=self.quantiles_model_non_monotonic_net,
                                                              module__dim_non_monotonic=len(self.numerics_non_monotonic) + sum(self.categoricals_out),
                                                              module__dim_monotonic=len(self.monotonic_constraints),
                                                              module__layers=layers,
                                                              module__dim_out=self.dim_y,
                                                              **kwargs)

    def _define_uncertainty_model(self,
                                  dim_certificates=64,
                                  max_epochs=250,
                                  lr=0.01,
                                  optimizer=torch.optim.Adam,
                                  **kwargs):
        '''
        Sets up Orthonormal Certificates model for estimating epistemic uncertainty of base model

        Parameters
        ----------
        dim_certificates : int, optional
            Number of linear mappings in Orthonormal Certificates model
        max_epochs : int, optional
            Number of epochs to train for (defaults to 250)
        lr : float, optional
            Learning rate for optimization (defaults to 0.01)
        optimizer : torch.optim.Optimizer
            PyTorch optimizer to use in training (defaults to torch.optim.Adam)

        Returns
        -------
        None
        '''
        self.uncertainty_model = OrthonormalCertificatesRegressor(OrthonormalCertificatesNet,
                                                                  max_epochs=max_epochs,
                                                                  lr=lr,
                                                                  optimizer=optimizer,
                                                                  iterator_train__shuffle=True,
                                                                  callbacks=[EarlyStopping(patience=max_epochs // 10)],
                                                                  module__dim_input=self.model_layers[-1] + len(self.monotonic_constraints),
                                                                  module__dim_certificates=dim_certificates)

    def _ingest(self, X, y):
        '''
        Read data frame on initialize and setup necessary attributes used later to define models

        Parameters
        ----------
        X : pandas.DataFrame
            Data frame that represents training input.
            No training is done during the initialization phase, but categories are counted and feature scalers are setup.
            So, if using a sample of the full dataset, it should be representative of the whole.

        Returns
        -------
        None
        '''
        self._prepare_categoricals(X)
        self._prepare_numerics(X, y)

    def _prepare_categoricals(self, X):
        '''
        Count unique values and create mappings for categorical features 
        
        Parameters
        ----------
        X : pandas.DataFrame
            Data frame that represents training input.

        Returns
        -------
        None
        '''
        # TODO: allow for specifying dim in and out
        self.categoricals = X[self.features].select_dtypes(include=['category', 'object']).columns.to_list()
        self.categoricals_in = []
        self.categoricals_out = []
        self.categoricals_maps = []
        for c in self.categoricals:
            if X[c].dtype == 'category':
                u = sorted(X[c].cat.categories.values)
            else:
                u = sorted(X[c].unique())
            self.categoricals_in.append(len(u) + 1)
            self.categoricals_out.append(min(max(round(np.sqrt(len(u))), 1), 256))
            self.categoricals_maps.append({v: i for i, v in enumerate(u, 1)})

    def _prepare_numerics(self, X, y):
        '''
        Create standard scalers for numeric features and target(s)

        Parameters
        ----------
        X : pandas.DataFrame
            Data frame that represents training input.
        y : pandas.DataFrame or numpy.array
            Data frame or array that represents training targets

        Returns
        -------
        None
        '''
        numerics = X[self.features].select_dtypes(include='number').columns
        self.numerics = numerics.to_list()
        self.numerics_non_monotonic = numerics.difference(self.monotonic_constraints.keys()).to_list()
        # X[self.numerics] = X[self.numerics]
        self.numerics_scaler = StandardScaler()
        self.numerics_scaler.fit(X[self.numerics])
        self.targets_scaler = StandardScaler()
        self.targets_scaler.fit(y.values)

    def _preprocess(self, X):
        '''
        Encode categorical features and scale or transform continuous features

        Parameters
        ----------
        X : pandas.DataFrame
            Data frame of training or prediction input

        Returns
        -------
        pandas.DataFrame
            Transformed data frame appropriate for training or prediction
        '''
        # TODO: figure out more memory efficient way to do this
        Xc = X.copy()
        for c, m in zip(self.categoricals, self.categoricals_maps):
            Xc[c] = Xc[c].map(m).astype('int').fillna(0)
        Xc[self.numerics] = self.numerics_scaler.transform(Xc[self.numerics])
        # TODO: Should S be a vector to support multi-target regression?
        for c, s in self.monotonic_constraints.items():
            Xc[c] = Xc[c] * s
        return Xc

    def _preprocess_targets(self, y):
        '''
        Scale target variable(s)

        Parameters
        ----------
        y : pandas.DataFrame or numpy.array
            Data frame of training or prediction input

        Returns
        -------
        pandas.DataFrame
            Transformed data frame appropriate for training or prediction
        '''
        # TODO: figure out more memory efficient way to do this
        return self.targets_scaler.transform(y.copy().values)

    def _postprocess_targets(self, predictions):
        '''
        Inverts the scaling of target variable(s)

        Parameters
        ----------
        predictions : numpy.ndarray
            Predicted values from model

        Returns
        -------
        numpy.ndarray
            Predictions in original scale of target variable(s)
        '''
        return self.targets_scaler.inverse_transform(predictions)

    def fit(self, X, y):
        '''
        Trains montonically constrained, Orthonormal Certificates, and Simultaneous Quantiles model

        Parameters
        ----------
        X : pandas.DataFrame
            Data frame of training input
        y : pandas.DataFrame or numpy.array
            Data frame or array of targets

        Returns
        -------
        None
        '''
        if not self.trained:
            self._setup(X, y)

        # TODO: Should make this flexible in case there aren't categoricals or non monotonic continuous features
        Xp = self._preprocess(X)
        Xd = {'X_monotonic': Xp[sorted(self.monotonic_constraints.keys())].values.astype('float32'),
              'X_categorical': Xp[self.categoricals].values.astype('int'),
              'X_non_monotonic': Xp[self.numerics_non_monotonic].values.astype('float32')}
        yp = self._preprocess_targets(y).astype('float32')
        print('*** Training expectation model ***')
        self.model.fit(Xd, yp)
        print('')
        print('*** Training epistemic uncertainty model ***')
        h = self.model.predict(Xd, last_hidden_layer=True)
        self.uncertainty_model.fit(np.concatenate([Xd['X_monotonic'], h], axis=1))
        print('')
        print('*** Training quantile prediction model ***')
        e = yp - self.model.predict(Xd)
        self.quantiles_model.fit(Xd, e)

        self.trained = True

    def predict(self, X):
        '''
        Predict target from monotonically constrained model

        Parameters
        ----------
        X : pandas.DataFrame
            Data frame with feature values

        Returns
        -------
        numpy.ndarray
            Predictions from base model
        '''
        Xp = self._preprocess(X)
        Xd = {'X_monotonic': Xp[sorted(self.monotonic_constraints.keys())].values.astype('float32'),
              'X_categorical': Xp[self.categoricals].values.astype('int'),
              'X_non_monotonic': Xp[self.numerics_non_monotonic].values.astype('float32')}
        predictions = self.model.predict(Xd)
        return self._postprocess_targets(predictions)

    def predict_quantile(self, X, q=0.5):
        '''
        Predict target quantile from Simultaneous Quantiles model

        Parameters
        ----------
        X : pandas.DataFrame
            Data frame with feature values
        q : float, optional
            Target quantile to predict (defaults to 0.5)

        Returns
        -------
        numpy.ndarray
            Predicted quantile
        '''
        Xp = self._preprocess(X)
        Xd = {'X_monotonic': Xp[sorted(self.monotonic_constraints.keys())].values.astype('float32'),
              'X_categorical': Xp[self.categoricals].values.astype('int'),
              'X_non_monotonic': Xp[self.numerics_non_monotonic].values.astype('float32')}
        predictions = self.model.predict(Xd)
        e = self.quantiles_model.predict(Xd, q=q)
        return self._postprocess_targets(predictions + e)

    def estimate_uncertainty(self, X):
        '''
        Estimate epistemic uncertainty of montonically constrained model

        Parameters
        ----------
        X : pandas.DataFrame
            Data frame with feature values

        Returns
        -------
        numpy.ndarray
            Ratio of epistemic uncertainty to max value seen in validation data
            (> 1 is higher uncertainty than seen in validation, < 1 is less)
        '''
        Xp = self._preprocess(X)
        Xd = {'X_monotonic': Xp[sorted(self.monotonic_constraints.keys())].values.astype('float32'),
              'X_categorical': Xp[self.categoricals].values.astype('int'),
              'X_non_monotonic': Xp[self.numerics_non_monotonic].values.astype('float32')}
        h = self.model.predict(Xd, last_hidden_layer=True)
        return self.uncertainty_model.scaled_predict(np.concatenate([Xd['X_monotonic'], h], axis=1))
