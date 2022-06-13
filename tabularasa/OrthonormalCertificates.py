import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class OrthonormalCertificates:

    def __init__(self,
                 dim_certificates=64,
                 epochs=500,
                 batch_size=128,
                 alpha=1
                 shuffle=True):
        self.dim_certificates = dim_certificates
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        # Add final loss

    def fit(self, X):
        self.certificates = nn.Linear(X.shape[1], self.dim_certificates)
        loader = DataLoader(TensorDataset(torch.tensor(X)),
                            shuffle=self.shuffle,
                            batch_size=self.batch_size)
        opt = torch.optim.Adam(self.certificates.parameters())

        for epoch in range(self.epochs):
            for xi in loader:
                opt.zero_grad()
                # Maybe have this support other loss functions than MSE
                error = self.certificates(xi[0]).pow(2).mean()
                penalty = (self.certificates.weight @ self.certificates.weight.t() -
                           torch.eye(self.dim_certificates)).pow(2).mean()
                (error + self.alpha * penalty).backward()
                opt.step()
        # Set and report on final loss
        # Do I need data standardization?  Optionally?  scikit moving away from this
        # Use a validation dataset to set the threshold?  This would allow me to create a score function

    def transform(self, X):
        # Should probably do some type checking here
        return self.certificates(torch.tensor(X)).pow(2).mean(1).detach().numpy()
