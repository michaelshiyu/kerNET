# -*- coding: utf-8 -*-
# torch 0.3.1

import torch
from torch.autograd import Variable
from kerlinear import kerLinear

import sys
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

    def add(self, component):
        if isinstance(component, kerLinear):
            setattr(self, 'comp'+str(self._comp_counter), component)
            self._comp_counter += 1

    def forward(self, x, X):
        i = 0
        ker_dims = [(
            getattr(self, 'comp'+str(i)).ker_dim
            ) for i in range(self._comp_counter)]
        out_dims = [(
            getattr(self, 'comp'+str(i)).out_dim
            ) for i in range(self._comp_counter)]

        # out_dims of all comps should be equal
        assert out_dims.count(out_dims[0])==len(out_dims)

        out_dim = out_dims[0]

        y = Variable(torch.FloatTensor(x.shape[0], out_dim).zero_())
        if x.is_cuda: y=y.cuda()
        # TODO: make X an attr of kerLinear. this may increase memory use but
        # makes things easier when fusing data coming from different domains.
        # if do that, need to change the following piece of code
        ###
        ker_dim = ker_dims[0]
        for i, X_batch in enumerate(K.get_batch(X, batch_size=ker_dim)):
            comp = getattr(self, 'comp'+str(i))
            setattr(comp, 'X', X_batch[0])
        ###

        for i in range(self._comp_counter):
            component = getattr(self, 'comp'+str(i))
            y = y.add(component.forward(x, component.X))
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
    linear_ens.add(kerLinear(ker_dim=2, out_dim=1, sigma=1, bias=True))
    linear_ens.add(kerLinear(ker_dim=1, out_dim=1, sigma=1, bias=False))
    linear_ens.comp1.weight.data = torch.FloatTensor([[1.5]])
    linear_ens.comp0.weight.data = torch.FloatTensor([[.5, .6]])
    linear_ens.comp0.bias.data = torch.FloatTensor([[2.5]])
    y = linear_ens(x, X)
    print(y)

    l = kerLinear(ker_dim=3, out_dim=1, sigma=1, bias=True)
    l.weight.data = torch.FloatTensor([[.5, .6, 1.5]])
    l.bias.data = torch.FloatTensor([[2.5]])
    y = l(x, X)
    print(y)
