# -*- coding: utf-8 -*-
# torch 0.4.0
import kernet.backend as K

import torch
from torch.autograd import Variable

torch.manual_seed(1234)

class kerLinear(torch.nn.Module):
    def __init__(self, X, out_dim, sigma, bias=True):
        """
        Building block for KN.
        A kernel linear layer first applies to input sample x a nonlinear map
        determined by the kernel function and the given random sample X:

            f : x -> R^n
            f : f(x_) -> (k(x_1, x_), k(x_2, x_), ..., k(x_n, x_)),

        where x_ \in x, k is a kernel function and X = {x_1, x_2, ..., x_n}.
        Then it linearly maps the image to some Euclidean space
        with dimension determined by the number of kernel machines in this
        layer. Currently only supports Gaussian kernel:
        k(x, y) = exp(-||x-y||_2^2 / (2 * sigma^2)). Currently only supports
        1darrays or 2darrays as input.

        Parameters
        ----------
        X : Tensor, shape (n_example, dim)
            On which this kernel machine is centered.

        out_dim : int
            The number of kernel machines in this layer.

        bias (optional) : bool
            If True, add a bias term to the linear combination.

        Attributes
        ----------
        """
        super(kerLinear, self).__init__()

        self.sigma = sigma
        if len(X.shape)==1: self.ker_dim=1 # does not modify the
        # dimension of X here as this will be done later in self.kerMap
        else: self.ker_dim=X.shape[0]
        self.out_dim = out_dim

        self.kerMap = K.kerMap
        self.linear = torch.nn.Linear(X.shape[0], out_dim, bias=bias)
        # TODO: customizable weight initializations

        # alias the parameters
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        # NOTE: save X as an attribute is useful when one wants to combine data
        # from multiple domains
        self.register_buffer('X', X)
        # remembers the initial state of X for kn._forward, use X.clone() to
        # break aliasing in case self.X is later modified
        self.register_buffer('X_init', X.clone())

    def forward(self, x, use_saved=False):
        """
        Parameters
        ----------

        x : Tensor, shape (batch_size, dim)

        use_saved (optional) : bool
            If True, use saved x_image. This is useful because in layerwise
            training, x_image is fixed throughout the training for the layer
            so there is no need for repeated computation.

        Returns
        -------
        y : Tensor, shape (batch_size, out_dim)
        """
        # TODO: use_saved is probably not going to help in batch mode since
        # for each batch x_image is different. unless save all batches in a dict
        # indexed by batch#
        # TODO: 3 modes: save, use_saved, do_nothing

        if not use_saved:
            # print(x, self.X)
            self.x_image = self.kerMap(x, self.X, self.sigma)
            # print(self.x_image)
        y = self.linear(self.x_image)

        return y

    def to_ensemble(self, batch_size):
        return K.to_ensemble(self, batch_size)

if __name__=='__main__':
    pass