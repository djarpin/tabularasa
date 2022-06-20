import torch
import numpy as np
from sklearn.base import RegressorMixin
from skorch import NeuralNet
from skorch.dataset import unpack_data
from skorch.utils import to_device
from skorch.utils import to_numpy
from skorch.utils import to_tensor


###############
# Loss function
###############


class OrthonormalCertificatesLoss(torch.nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', alpha=1) -> None:
        super(OrthonormalCertificatesLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        error = input.mean()
        penalty = (weights @ weights.t() - torch.eye(weights.size(0))).pow(2).mean()
        return error + self.alpha * penalty


####################
# Network definition
####################


class OrthonormalCertificatesNet(torch.nn.Module):

    def __init__(self, dim_input, dim_certificates=64):
        super().__init__()
        self.certificates = torch.nn.Linear(dim_input, dim_certificates)

    def forward(self, X):
        return self.certificates(X).pow(2).mean(1)


##################
# Regressor object
##################


class OrthonormalCertificatesRegressor(NeuralNet, RegressorMixin):

    def __init__(self, module, *args, criterion=OrthonormalCertificatesLoss, **kwargs):
        super(OrthonormalCertificatesRegressor, self).__init__(module, *args, criterion=criterion, **kwargs)

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
        return self.criterion_(y_pred, y_true, weights=self.module_.certificates.weight)

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
        with torch.no_grad():
            y_pred = self.infer(Xi, **fit_params)
            percentiles = torch.arange(0.01, 1.01, 0.01)
            self._validation_percentiles = torch.quantile(y_pred, percentiles).numpy()
            loss = self.get_loss(y_pred, yi, X=Xi, training=False)
        return {
            'loss': loss,
            'y_pred': y_pred,
        }

    def percentile_predict(self, X):
        p = self.predict(X)
        return np.searchsorted(self._validation_percentiles, p)
