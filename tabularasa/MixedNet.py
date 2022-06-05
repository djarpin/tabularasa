import torch.nn as nn
from tabularasa.gumnn.MultidimensionnalMonotonicNN import SlowDMonotonicNN


class MixedNet(nn.Module):

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
        self.umnn = SlowDMonotonicNN(dim_monotonic,
                                     dim_non_monotonic,
                                     layers,
                                     dim_out,
                                     integration_steps,
                                     device)

    def forward(self, X_monotonic, X_non_monotonic):
        h = self.non_monotonic_net(X_non_monotonic)
        return self.umnn(X_monotonic, h)
