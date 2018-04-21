# -*- coding: utf-8 -*-
# torch 0.3.1

import torch
from torch.autograd import Variable

import sys
sys.path.append('../kernet/layers')
from kerlinear import kerLinear
sys.path.append('../kernet')
import backend as K

torch.manual_seed(1234)

class _ensemble(torch.nn.Module):
    def __init__(self):
        super(_ensemble, self).__init__()
        self._comp_counter = 0
    def add(self,):
        raise NotImplementedError('Must be implemented by subclass.')
    def forward(self,):
        raise NotImplementedError('Must be implemented by subclass.')

class kerLinearEnsemble(_ensemble):
    # NOTE: only one of the components need a bias
    # TODO: another issue is that initializing each component separately results
    # in different initial weights compared to normal kerLinear
    def __init__(self):
        super(kerLinearEnsemble, self).__init__()
        self.X = self._X() # generator for X's

    def _X(self):
        """
        Generate an iterable of X's from each component kerLinear layer in this
        ensemble.
        """
        for i in range(self._comp_counter):
            comp = getattr(self, 'comp'+str(i))
            yield comp.X

    def add(self, component):
        # assert isinstance(component, kerLinear) # BUG
        setattr(self, 'comp'+str(self._comp_counter), component)
        self._comp_counter += 1
        self.sigma = component.sigma # TODO: allow components to have different
        # sigma?

    def forward(self, x):
        out_dims = [(
            getattr(self, 'comp'+str(i)).out_dim
            ) for i in range(self._comp_counter)]
        # out_dims of all comps should be equal
        assert out_dims.count(out_dims[0])==len(out_dims)

        out_dim = out_dims[0]

        y = Variable(torch.FloatTensor(x.shape[0], out_dim).zero_())
        if x.is_cuda: y=y.cuda()

        for i in range(self._comp_counter):
            component = getattr(self, 'comp'+str(i))
            y = y.add(component.forward(x))
        self.out_dim = out_dim
        return y

"""
def baseMLKN.to_ensemble(self, layer, X_batch_size):
    assert isinstance(layer, kerLinear)
    ensemble = _kerLinearEnsemble()
    n_batch = layer.ker_dim // X_batch_size
    last_batch = layer.ker_dim % X_batch_size

    for i in range(n_batch):
        ensemble.add(kerLinear(
            ker_dim=batch_size,
            out_dim=layer.out_dim,
            sigma=layer.sigma,
            bias=layer.bias
            ))
        ensemble._comp_counter += 1
    if last_batch:
        ensemble.add(kerLinear(
            ker_dim=last_batch,
            out_dim=layer.out_dim,
            sigma=layer.sigma,
            bias=layer.bias
            ))
        ensemble._comp_counter += 1
"""
if __name__=='__main__':
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor

    x = Variable(torch.FloatTensor([[0, 7], [1, 2]]).type(dtype))
    X = Variable(torch.FloatTensor([[1, 2], [3, 4], [5, 6]]).type(dtype))
    y = Variable(torch.FloatTensor([[.3], [.9]]).type(dtype))

    linear_ens = kerLinearEnsemble()
    linear_ens.add(kerLinear(X[:2], out_dim=1, sigma=1, bias=True))
    linear_ens.add(kerLinear(X[2:], out_dim=1, sigma=1, bias=False))
    linear_ens.comp1.weight.data = torch.FloatTensor([[1.5]])
    linear_ens.comp0.weight.data = torch.FloatTensor([[.5, .6]])
    linear_ens.comp0.bias.data = torch.FloatTensor([[2.5]])
    y = linear_ens(x)
    print(y)

    l = kerLinear(X, out_dim=1, sigma=1, bias=True)
    l.weight.data = torch.FloatTensor([[.5, .6, 1.5]])
    l.bias.data = torch.FloatTensor([[2.5]])
    y = l(x)
    print(y)
