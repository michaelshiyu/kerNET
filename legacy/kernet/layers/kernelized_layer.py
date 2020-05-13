#!/usr/bin/env python


import kernet.backend as K
import torch

# TODO tests
# TODO documentation is behind code

class _kernelizedLayer(torch.nn.Module):
    """
    Base class for all kernel layers.

    Need this base class since KN requires special treatment in greedy learning 
    (update X, etc.) so it would be convenient to have a base class for all KN 
    layers so that we can quickly differentiate a KN layer from an NN layer at 
    runtime and adjust settings such as update X accordingly.
    """
    def __init__(self):
        super().__init__()

class kFullyConnected(_kernelizedLayer): 
    def __init__(self, X, n_out, kernel='gaussian', sigma=1, bias=True, trainable_X=False):
        """
        A kernelized layer that is fully-connected with the previous layer (or input vector).
        This layer first applies to input x a nonlinear map
        determined by the kernel function and the given random sample X:

            f : x -> R^n
            f : f(x_) -> (k(x_1, x_), k(x_2, x_), ..., k(x_n, x_)),

        where x_ \in x, k is a kernel function and X = {x_1, x_2, ..., x_n}.
        Then it linearly maps the image to some Euclidean space
        with dimension determined by the number of kernel machines in this
        layer. Currently only supports Gaussian kernel:
        k(x, y) = exp(-||x-y||_2^2 / (2 * sigma^2)). 

        Currently checked 1darrays or 2darrays as input. TODO check for d>2

        Parameters
        ----------
        X : Tensor, shape (n_example, dim)
            Centers on which the kernel machines are centered.

        n_out : int
            The number of kernel machines in this layer.

        bias (optional) : bool
            If True, add a bias term to the linear combination.

        trainable_X (optional) : bool
            Whether to treat the centers as trainable parameters. If so, just need to 
            pass a few randomly chosen examples from the training set (or just 
            random vectors) and the model will adapt its centers by itself. 
            Will potentially sacrifice some performance for speed.

        """
        super().__init__()
        self.kernel = kernel.lower()
        assert self.kernel in ['gaussian']

        self.n_in = X.shape[0]
        self.n_out = n_out

        self.phi = K.Phi(sigma=sigma, kernel=self.kernel)
        self.linear = torch.nn.Linear(self.n_in, self.n_out, bias=bias)
        # TODO: customizable weight initializations

        # alias the parameters
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        if trainable_X:
            self.trainable_X = True

            # this adds a copy of X to the trainable parameters of the layer
            self.X = torch.nn.Parameter(X.clone().detach().requires_grad_(True))

        else:
            # save X as an attribute is useful when one wants to combine data
            # from multiple domains
            self.register_buffer('X', X)

            # remembers the initial state of X for kn._forward, use X.clone() to
            # break aliasing in case self.X is later modified
            self.register_buffer('X_init', X.clone())

    def forward(self, x):
        """
        Parameters
        ----------

        x : Tensor, shape (batch_size, dim)

        Returns
        -------
        y : Tensor, shape (batch_size, n_out)
        """

        return self.linear(self.phi(x, X=self.X))

    def to_ensemble(self, batch_size):
        return K.to_ensemble(self, batch_size)

class _kernelizedLayerMultiComp(_kernelizedLayer):
    """
    Base class for kernel layers with multiple components.
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

    def add_comp(self, component):
        """
        Add a component to this instance.
        """
        assert isinstance(component, kFullyConnected)
        self.add_module('comp'+str(self._comp_counter), component)
        self._comp_counter += 1
        
    def forward(self):
        raise NotImplementedError('Must be implemented by subclass.')

class kFullyConnectedEnsemble(_kernelizedLayerMultiComp):
    """
    Essentially the same layer as kFullyConnected. But kFullyConnectedEnsemble breaks the set of centers X
    into a few, say, n, smaller chunks and initialize n kFullyConnectedEnsembles in 
    parallel.

    There is a handy function in backend called to_ensemble
    that when passed with a kFullyConnected instance, returns a 
    replica of it as a kFullyConnectedEnsemble instance with exactly the same weights.

    Using this replica of the original kFullyConnected would result in exactly same
    results and everything. The difference is that kFullyConnectedEnsemble allows you 
    to fit larger datasets into memory, which can be impossible to do with a 
    naive implementation, such as kFullyConnected. But it could run slower than the 
    kFullyConnected model, depending on how small the chunks you partitioned X
    into are and your computing environment (for example, CPU or GPU).
    """

    def add_comp(self, component):
        """
        Override _kernelizedLayerMultiComp.add because kFullyConnectedEnsemble has extra
        constraints. Namely, the components, as parts of what originally was
        a kFullyConnected layer, should have identical kernels and output dimensions.
        """
        assert isinstance(component, kFullyConnected)
        self.add_module('comp'+str(self._comp_counter), component)
        self._comp_counter += 1
        # TODO: allow components to have different sigma? does that make sense?
        if self._comp_counter==1:
            self.phi = component.phi
        else:
            pass
            # assert component.phi==self.phi # FIXME this is incorrect for KN layers since component.phi.X is not layer.X. This is fine for computing actual or ideal kmtrx since X is overidden there, but what might go wrong? How about returning a generator of phi's when calling self._phi?

        if self._comp_counter == 1:
            self.n_out = component.n_out
        elif self._comp_counter > 1:
            # n_out of all components should be equal
            assert self.n_out == component.n_out
            

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

        y = x.new_zeros(x.shape[0], self.n_out)

        for i in range(self._comp_counter):
            component = getattr(self, 'comp'+str(i))
            y = y.add(component(x))
        return y

class kFullyConnectedStack(_kernelizedLayerMultiComp):
    """
    TODO deprecate this class? this can be implemented in a purely PyTorch (no need for kerNET) way

    Stack a few kFullyConnected layers into one layer and train them together with
    backpropagation. This should be only used as the first layer of your 
    greedily-trained KN.

    The first element added to the stack should be the layer closest to input.
    """
    # TODO option to convert kFullyConnectedStack to kFullyConnectedEnsemble
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