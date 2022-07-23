import numpy as np
import torch
import torch.nn as nn
from sklearn.base import RegressorMixin
from skorch import NeuralNet
from skorch.dataset import unpack_data
from skorch.utils import to_device
from skorch.utils import to_numpy
from tabularasa.gumnn.MultidimensionnalMonotonicNN import SlowDMonotonicNN


####################
# Network definition
####################


class MixedMonotonicNet(nn.Module):

    def __init__(self,
                 non_monotonic_net,
                 dim_non_monotonic,
                 dim_monotonic,
                 layers=[512, 512, 64],
                 dim_out=1,
                 integration_steps=50,
                 device='cpu'):
        super().__init__()
        self.non_monotonic_net = non_monotonic_net
        self.monotonic_net = SlowDMonotonicNN(dim_monotonic,
                                              dim_non_monotonic,
                                              layers,
                                              dim_out,
                                              integration_steps,
                                              device)

    def forward(self, X_monotonic, X_non_monotonic, last_hidden_layer=False):
        h = self.non_monotonic_net(X_non_monotonic)
        return self.monotonic_net(X_monotonic, h, last_hidden_layer)


##################
# Regressor object
##################


class MixedMonotonicRegressor(NeuralNet, RegressorMixin):

    def __init__(self, module, *args, criterion=nn.MSELoss, **kwargs):
        super(MixedMonotonicRegressor, self).__init__(module, *args, criterion=criterion, **kwargs)

    def evaluation_step(self, batch, training=False, last_hidden_layer=False):
        """Perform a forward step to produce the output used for
        prediction and scoring.
        Therefore, the module is set to evaluation mode by default
        beforehand which can be overridden to re-enable features
        like dropout by setting ``training=True``.
        Parameters
        ----------
        batch
          A single batch returned by the data loader.
        training : bool (default=False)
          Whether to set the module to train mode or not.
        Returns
        -------
        y_infer
          The prediction generated by the module.
        """
        self.check_is_fitted()
        Xi, _ = unpack_data(batch)
        with torch.set_grad_enabled(training):
            self._set_training(training)
            return self.infer(Xi, last_hidden_layer=last_hidden_layer)

    def forward_iter(self, X, training=False, device='cpu', last_hidden_layer=False):
        """Yield outputs of module forward calls on each batch of data.
        The storage device of the yielded tensors is determined
        by the ``device`` parameter.
        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        training : bool (default=False)
          Whether to set the module to train mode or not.
        device : string (default='cpu')
          The device to store each inference result on.
          This defaults to CPU memory since there is genereally
          more memory available there. For performance reasons
          this might be changed to a specific CUDA device,
          e.g. 'cuda:0'.
        Yields
        ------
        yp : torch tensor
          Result from a forward call on an individual batch.
        """
        dataset = self.get_dataset(X)
        iterator = self.get_iterator(dataset, training=training)
        for batch in iterator:
            yp = self.evaluation_step(batch, training=training, last_hidden_layer=last_hidden_layer)
            yield to_device(yp, device=device)

    def predict_proba(self, X, last_hidden_layer=False):
        """Return the output of the module's forward method as a numpy
        array.
        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.
        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        Returns
        -------
        y_proba : numpy ndarray
        """
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, training=False, last_hidden_layer=last_hidden_layer):
            yp = yp[0] if isinstance(yp, tuple) else yp
            yp = nonlin(yp)
            y_probas.append(to_numpy(yp))
        y_proba = np.concatenate(y_probas, 0)
        return y_proba

    def predict(self, X, last_hidden_layer=False):
        """Where applicable, return class labels for samples in X.
        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.
        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        Returns
        -------
        y_pred : numpy ndarray
        """
        return self.predict_proba(X, last_hidden_layer=last_hidden_layer)
