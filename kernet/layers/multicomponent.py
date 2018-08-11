# -*- coding: utf-8 -*-
# updated documentation Aug10_2018

import kernet.backend as K
from kernet.layers.kerlinear import kerLinear

import torch
from torch.autograd import Variable

torch.manual_seed(1234)

class _multiComponent(torch.nn.Module):
    """
    Base class for layer with multiple components.
    """
    def __init__(self):
        super().__init__()
        self._comp_counter = 0

        # generator objects, contain the X, weight and bias of each component
        self.X = self._X() 
        self.weight = self._weight()
        self.bias = self._bias()

    def _X(self):
        """
        Generate X of each component in the order in which the components
        were added.
        """
        for i in range(self._comp_counter):
            comp = getattr(self, 'comp'+str(i))
            yield comp.X

    def _weight(self):
        """
        Generate weights of each component in the order in which the components
        were added.
        """
        for i in range(self._comp_counter):
            comp = getattr(self, 'comp'+str(i))
            yield comp.weight

    def _bias(self):
        """
        Generate bias of each component in the order in which the components
        were added.
        """
        for i in range(self._comp_counter):
            comp = getattr(self, 'comp'+str(i))
            yield comp.bias

    def add(self, component):
        """
        Add a component to this instance.
        """
        assert isinstance(component, kerLinear)
        self.add_module('comp'+str(self._comp_counter), component)
        self._comp_counter += 1
        
    def forward(self):
        raise NotImplementedError('Must be implemented by subclass.')

class kerLinearEnsemble(_multiComponent):
    """
    Essentially the same layer as kerLinear. But kerLinearEnsemble breaks X
    into a few, say, n, smaller chunks and initialize n kerLinearEnsembles in 
    parallel.

    There is a handy function in backend called to_ensemble
    that when passed with a kerLinear instance, returns a 
    replica of it as a kerLinearEnsemble instance with exactly the same weights.

    Using this replica of the original kerLinear would result in exactly same
    results and everything. The difference is that kerLinearEnsemble allows you 
    to fit larger datasets into memory, which can be impossible to do with a 
    naive implementation, such as kerLinear. But it could run slower than the 
    kerLinear model, depending on how small the chunks you partitioned X
    into are and your computing environment (for example, CPU or GPU).
    """

    def add(self, component):
        """
        Override _multiComponent.add because kerLinearEnsemble has extra
        constraints. Namely, the components, as parts of what originally was
        a kerLinear layer, should have identical kernels and output dimensions.
        """
        assert isinstance(component, kerLinear)
        self.add_module('comp'+str(self._comp_counter), component)
        self._comp_counter += 1
        # TODO: allow components to have different sigma? does that make sense?
        self.sigma = component.sigma 

        if self._comp_counter == 1:
            self.out_dim = component.out_dim
        elif self._comp_counter > 1:
            # out_dim of all components should be equal
            assert self.out_dim == component.out_dim
            

    def forward(self, x):
        """
        Feed x into the layer and get output.

        Parameters
        ----------

        x : Tensor, shape (batch_size, dim)
            Data to be evaluated.
        """
        # FIXME: under shuffle mode, fit gives different
        # results if substitute normal layers with ensemble layers, checked that
        # the randperm vectors in K.rand_shuffle are different in two modes, why?

        y = x.new_zeros(x.shape[0], self.out_dim)

        for i in range(self._comp_counter):
            component = getattr(self, 'comp'+str(i))
            y = y.add(component(x))
        return y

class kerLinearStack(_multiComponent):
    """
    Stack a few kerLinear layers into one layer and train them together with
    backpropagation. This should be only used as the first layer of your 
    greedily-trained KN.

    The first element added to the stack should be the layer closest to input.
    """
    # TODO option to convert kerLinearStack to kerLinearEnsemble
    def forward(self, x, upto=None):
        """
        Feedforward upto and including the 'upto'th layer in the stack.

        Parameters
        ----------

        x : Tensor, shape (batch_size, dim)
            Data to be evaluated.

        upto (optional) : int
            Upto and including this layer, x will be fed into the stack and 
            evaluated.
        """
        if upto is not None: 
        # cannot use 'if upto' here since it is 0-indexed
        # and layer0 is the first layer
            assert 0<=upto<=self._comp_counter
            upto += 1
        else: upto = self._comp_counter # evaluate all layers instead

        y_previous = x
        for i in range(upto):
            component = getattr(self, 'comp'+str(i))
            y = component(y_previous)
            y_previous = y

        return y

    def to_ensemble_(self, batch_size):
        """
        Convert layers in the stack into equivalent ensemble layers.
        Note that this is an in-place function.

        Parameters
        ----------
        batch_size : int
            Size of each component layer in the ensemble. One batch_size for 
            all layers in the stack.
        """
        for i in range(self._comp_counter):
            component = getattr(self, 'comp'+str(i))
            setattr(self, 'comp'+str(i), K.to_ensemble(component, batch_size))
            
if __name__=='__main__':
    pass