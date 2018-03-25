# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from .. import backend as

class kerLinear(torch.nn.Module):
    def __init__(self, out_dim):
        """
        Building block for MLKN.
        A kernel linear layer first applies to input a nonlinear map
        determined by the kernel function and the given random sample, then
        linearly map the image to a new representation in some Euclidean space
        with dimension determined by the number of kernel machines in this
        layer.

        Parameters
        ----------
        out_dim : int
            The number of kernel machines in this layer.

        Returns
        -------

        Attributes
        ----------
        """
