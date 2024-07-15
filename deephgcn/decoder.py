import torch.nn as nn
import torch
import math
import numpy as np 
import torch.nn.functional as F
import geoopt

class FermiDiracDecoder(nn.Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp(((dist - self.r) / self.t).clamp_max(50.)) + 1.0)
        return probs