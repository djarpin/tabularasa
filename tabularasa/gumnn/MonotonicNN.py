import torch
import torch.nn as nn
from .ParallelNeuralIntegral import ParallelNeuralIntegral


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())
        self.inner_net = nn.Sequential(*self.net[:-2])
        self.net = nn.Sequential(*self.net)

    def to(self, device):
        self.net.to(device)

    def forward(self, x, h, last_hidden_layer=False):
        # (darpin) Modified for orthonormal certificates
        if last_hidden_layer:
            return self.inner_net(torch.cat((x, h), 1)) + 1.
        return self.net(torch.cat((x, h), 1)) + 1.

class MonotonicNN(nn.Module):
    def __init__(self, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        super(MonotonicNN, self).__init__()
        self.integrand = IntegrandNN(in_d, hidden_layers)
        self.net = []
        hs = [in_d-1] + hidden_layers + [2]
        for h0, h1 in zip(hs[:-1], hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        # It will output the scaling and offset factors.
        self.net = nn.Sequential(*self.net)
        self.device = dev
        self.nb_steps = nb_steps
        self.to(dev)

    def to(self, device):
        self.net.to(device)
        self.integrand.to(device)

    '''
    The forward procedure takes as input x which is the variable for which the integration must be made, h is just other conditionning variables.
    '''
    def forward(self, x, h, last_hidden_layer=False):
        x0 = torch.zeros(x.shape).to(self.device)
        out = self.net(h)
        offset = out[:, [0]]
        scaling = torch.exp(out[:, [1]])
        # (darpin) Modified for orthonormal certificates
        if last_hidden_layer:
            return self.integrand(x, h, last_hidden_layer)
        return scaling*ParallelNeuralIntegral.apply(x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps) + offset
