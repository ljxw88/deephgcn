import torch
import geoopt
import torch.nn.functional as F
from geoopt.manifolds.stereographic.math import _project
import math
from .layers import MobiusLinear, MobiusLinearPlus, MobiusLinearFast

class NetEuclidean(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(NetEuclidean, self).__init__()
        self.layer = torch.nn.Linear(n_feature, n_output)

    def forward(self, x):
        x = (self.layer.weight @ x.T).T
        x = x + self.layer.bias
        return x

class NetHype(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(NetHype, self).__init__()
        self.layer = MobiusLinear(n_feature, n_output)

    def forward(self, x):
        x = self.layer(x)
        return x

class NetHypePlus(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(NetHypePlus, self).__init__()
        self.layer = MobiusLinearPlus(n_feature, n_output)

    def forward(self, x):
        x = self.layer(x)
        return x

class NetHypeFast(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(NetHypeFast, self).__init__()
        self.layer = MobiusLinearFast(n_feature, n_output)

    def forward(self, x):
        x = self.layer(x)
        return x