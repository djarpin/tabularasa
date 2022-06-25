import numpy as np
from sklearn.base import RegressorMixin
import torch
from skorch import NeuralNet
from skorch.dataset import unpack_data
from skorch.utils import to_device
from skorch.utils import to_numpy
from skorch.utils import to_tensor
from tabularasa.MixedMonotonic import MixedMonotonicNet
from tabularasa.gumnn.MultidimensionnalMonotonicNN import SlowDMonotonicNN


###############
# Loss function
###############


class SimultaneousQuantilesLoss(torch.nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(SimultaneousQuantilesLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
        diff = input - target
        threshold = (diff.ge(0).float() - qs).detach()
        return (threshold * diff).mean()


#####################
# Network definitions
#####################


class SimultaneousQuantilesNet(MixedMonotonicNet):

    def __init__(self,
                 non_monotonic_net,
                 dim_non_monotonic,
                 dim_monotonic,
                 layers=[512, 512, 64],
                 dim_out=1,
                 integration_steps=50,
                 device='cpu'):
        super().__init__(non_monotonic_net,
                         dim_non_monotonic,
                         dim_monotonic,
                         layers,
                         dim_out,
                         integration_steps,
                         device)
        self.non_monotonic_net = non_monotonic_net
        self.umnn = SlowDMonotonicNN(1,
                                     dim_non_monotonic,
                                     layers,
                                     dim_out,
                                     integration_steps,
                                     device)

    def forward(self, X_non_monotonic, qs, last_hidden_layer=False):
        h = self.non_monotonic_net(X_non_monotonic)
        return self.umnn(qs, h, last_hidden_layer)


class SimultaneousQuantilesMixedMonotonicNet(MixedMonotonicNet):

    def __init__(self,
                 non_monotonic_net,
                 dim_non_monotonic,
                 dim_monotonic,
                 layers=[512, 512, 64],
                 dim_out=1,
                 integration_steps=50,
                 device='cpu'):
        super().__init__(non_monotonic_net,
                         dim_non_monotonic,
                         dim_monotonic,
                         layers,
                         dim_out,
                         integration_steps,
                         device)
        self.non_monotonic_net = non_monotonic_net
        self.umnn = SlowDMonotonicNN(dim_monotonic + 1,
                                     dim_non_monotonic,
                                     layers,
                                     dim_out,
                                     integration_steps,
                                     device)

    def forward(self, X_monotonic, X_non_monotonic, qs, last_hidden_layer=False):
        h = self.non_monotonic_net(X_non_monotonic)
        return self.umnn(torch.cat([X_monotonic, qs], 1), h, last_hidden_layer)


##################
# Regressor object
##################


class SimultaneousQuantilesRegressor(NeuralNet, RegressorMixin):

    def __init__(self, module, *args, criterion=SimultaneousQuantilesLoss, **kwargs):
        super(SimultaneousQuantilesRegressor, self).__init__(module, *args, criterion=criterion, **kwargs)

    def get_loss(self, y_pred, y_true, X, training=False):
        """Return the loss for this batch.
        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values
        y_true : torch tensor
          True target values.
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
          Whether train mode should be used or not.
        """
        y_true = to_tensor(y_true, device=self.device)
        return self.criterion_(y_pred, y_true, X['qs'])

    def train_step_single(self, batch, **fit_params):
        """Compute y_pred, loss value, and update net's gradients.
        The module is set to be in train mode (e.g. dropout is
        applied).
        Parameters
        ----------
        batch
          A single batch returned by the data loader.
        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.
        Returns
        -------
        step : dict
          A dictionary ``{'loss': loss, 'y_pred': y_pred}``, where the
          float ``loss`` is the result of the loss function and
          ``y_pred`` the prediction generated by the PyTorch module.
        """
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        Xi['qs'] = torch.rand(yi.size(0), 1)
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        loss.backward()
        return {
            'loss': loss,
            'y_pred': y_pred,
        }

    def validation_step(self, batch, **fit_params):
        """Perform a forward step using batched data and return the
        resulting loss.
        The module is set to be in evaluation mode (e.g. dropout is
        not applied).
        Parameters
        ----------
        batch
          A single batch returned by the data loader.
        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.
        """
        self._set_training(False)
        Xi, yi = unpack_data(batch)
        Xi['qs'] = torch.rand(yi.size(0), 1)
        with torch.no_grad():
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=False)
        return {
            'loss': loss,
            'y_pred': y_pred,
        }

    def evaluation_step(self, batch, training=False, q=0.5, last_hidden_layer=False):
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
        Xi['qs'] = torch.full((list(Xi.values())[0].size(0), 1), q)
        with torch.set_grad_enabled(training):
            self._set_training(training)
            return self.infer(Xi, last_hidden_layer=last_hidden_layer)

    def forward_iter(self, X, training=False, device='cpu', q=0.5, last_hidden_layer=False):
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
            yp = self.evaluation_step(batch, training=training, q=q, last_hidden_layer=last_hidden_layer)
            yield to_device(yp, device=device)

    def predict_proba(self, X, q=0.5, last_hidden_layer=False):
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
        for yp in self.forward_iter(X, training=False, q=q, last_hidden_layer=last_hidden_layer):
            yp = yp[0] if isinstance(yp, tuple) else yp
            yp = nonlin(yp)
            y_probas.append(to_numpy(yp))
        y_proba = np.concatenate(y_probas, 0)
        return y_proba

    def predict(self, X, q=0.5, last_hidden_layer=False):
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
        return self.predict_proba(X, q=q, last_hidden_layer=last_hidden_layer)
