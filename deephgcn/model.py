import os, sys, inspect
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from itertools import dropwhile
import torch.nn as nn
import torch
import math
import numpy as np 
import torch.nn.functional as F
import geoopt
from geooptplus import PoincareBall
from geoopt import ManifoldParameter, ManifoldTensor
from layers import MobiusLinear, MobiusLinearPlus, MobiusLinearFast


class HyperbolicLinearFast(nn.Module):
    def __init__(self, c, in_features, out_features, dropout, use_bias=True, nonlin=None):
        super().__init__()
        self.use_bias = use_bias
        self.c = c
        self.in_features = in_features
        self.out_features = out_features
        self.layer = MobiusLinearFast(in_features, out_features, nonlin=None, c=self.c, dropout=dropout)
        self.act_fn = nonlin

    def forward(self, x):
        x = self.layer(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

class HyperbolicGraphConvolution(nn.Module):
    def __init__(self, manifold, in_features, out_features, dropout, layer_num, act_fn=None, final=False, params=[]):
        super(HyperbolicGraphConvolution, self).__init__() 

        self.use_bias = True
        self.learnable_c = False

        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.act_fn = act_fn
        self.layer = HyperbolicLinearFast(self.manifold.c, in_features, out_features, dropout=dropout, nonlin=act_fn)
        self.dropout = dropout
        self.use_att = False
        self.final = final
        
        self.layer_num = layer_num
        self.alpha = params[0]
        self.beta = params[1]

        self.final_agg = False

    def forward(self, input, adj, h_init):
        theta = math.log(self.beta/self.layer_num + 1)
        support = input

        # hyperbolic neighbourhood aggregation
        support = self.manifold.weighted_midpoint_spmm(support, weights=adj)

        # residual connection
        weight_residual = torch.stack([torch.tensor(1-self.alpha), torch.tensor(self.alpha)]).to(input)
        support = self.manifold.weighted_midpoint(torch.stack([support, h_init], dim = 1), \
            weights = weight_residual, reducedim = [1])
        
        if self.final:
            return support
        
        # saved residual
        support_res = support

        # hyperbolic linear
        support = self.layer(support)

        # weight alignment
        if self.final_agg:
            weight_alignment = torch.stack([torch.tensor(theta), torch.tensor(1-theta)]).to(input)
            support = self.manifold.weighted_midpoint(torch.stack([support, support_res], dim = 1), \
                weights = weight_alignment, reducedim = [1])

        return support

class HGCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, act_fn, c, params):
        super(HGCN, self).__init__()
        self.c = c
        self.manifold = PoincareBall(c=self.c)
        self.convs = nn.ModuleList()
        self.nlayers = nlayers
        self.dropout = dropout
        self.fcs = nn.ModuleList()
        self.fcs.append(MobiusLinearFast(nfeat, nhidden, c=self.c, dropout=dropout, nonlin=act_fn))
        self.fcs.append(MobiusLinearFast(nhidden, nclass, c=self.c, dropout=dropout, nonlin=None))
        for ln in range(self.nlayers):
            final_layer = False
            if ln == self.nlayers - 1:
                final_layer = True
            self.convs.append(HyperbolicGraphConvolution(self.manifold, nhidden, nhidden, dropout, \
                                                         layer_num=ln+1, act_fn=act_fn, final=final_layer, params=params))
        
        self.params_convs = list(self.convs.parameters())
        self.params_fcs = list(self.fcs.parameters())

    def forward(self, x, adj):
        _layers = []
        layer_inner = x
        layer_inner = self.fcs[0](layer_inner)
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = con(layer_inner, adj, _layers[0])
            _layers.append(layer_inner)
        
        support = self.fcs[-1](layer_inner)
        # output = self.manifold.logmap0(output)
        output = F.log_softmax(support, dim=1)
        return output, []

    def encode(self, x, adj):
        return self.forward(x, adj)
    
    def fetchEmbeddings(self, x, adj):
        '''To Implememt'''
        return None


if __name__ == '__main__':
    pass
