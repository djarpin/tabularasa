import torch
from torch import nn
from torch.data.utils import TensorDataset, DataLoader


class OrthonormalCertificates:

    def __init__(self,
                 dim_certificates=64,
                 epochs=500,
                 batch_size=128,
                 shuffle=True):
        self.dim_certificates = dim_certificates
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle

    def fit(self, X):
        certificates = nn.Linear(X.size(1), self.dim_certificates)
        loader = DataLoader(TensorDataset(torch.tensor(X)),
                            shuffle=self.shuffle,
                            batch_size=self.batch_size)
        opt = torch.optim.Adam(certificates.parameters())

        for epoch in range(self.epochs):
            for xi in loader:
                opt.zero_grad()
                error = certificates(xi[0]).pow(2).mean()
                penalty = (certificates.weight @ certificates.weight.t() - 
                           torch.eye(self.dim_certificates)).pow(2).mean()
                (error + penalty).backward()
                opt.step()

    def transform(X):
        # Should probably do some type checking here
        return self.certificates(torch.tensor(X)).pow(2).mean(1)
