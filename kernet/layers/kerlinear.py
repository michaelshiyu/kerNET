# -*- coding: utf-8 -*-
# torch 0.3.1

import torch
from torch.autograd import Variable

import backend as K

torch.manual_seed(1234)

class kerLinear(torch.nn.Module):
    def __init__(self, ker_dim, out_dim, sigma, bias=True):
        """
        Building block for MLKN.
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
        out_dim : int
            The number of kernel machines in this layer.
        ker_dim : int
            Dimension of the codomain of f, or, equivalently, the size of the
            random sample X.
        bias (optional) : bool
            If True, add a bias term to the linear combination.

        Attributes
        ----------
        """
        super(kerLinear, self).__init__()

        self.sigma = sigma
        self.ker_dim = ker_dim

        self.kerMap = K.kerMap
        self.linear = torch.nn.Linear(ker_dim, out_dim, bias=bias)
        # TODO: customizable weight initializations

        # alias the parameters
        self.weight = self.linear.weight
        self.bias = self.linear.bias

    def forward(self, x, X):
        """
        Parameters
        ----------

        x : Tensor, shape (batch_size, dim)

        X : Tensor, shape (n_example, dim)

        Returns
        -------
        y : Tensor, shape (batch_size, out_dim)
        """
        if len(X.shape)==1: assert self.ker_dim==1 # does not modify the
        # dimension of X here as this will be done later in self.kerMap
        else: assert self.ker_dim==X.shape[0]

        x_image = self.kerMap(x, X, self.sigma)
        """
        print('x_image', x_image)
        """
        y = self.linear(x_image)
        return y

if __name__=='__main__':

    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor

    x = Variable(torch.FloatTensor([[0, 7], [1, 2]]).type(dtype))
    X = Variable(torch.FloatTensor([[1, 2], [3, 4], [5, 6]]).type(dtype))
    y = Variable(torch.FloatTensor([[.3], [.9]]).type(dtype))

    l = kerLinear(ker_dim=3, out_dim=1, sigma=1, bias=True)

    y_pred = l(x=x, X=X)
    print('y_pred', y_pred)
    print('weight', l.linear.weight)
    print('bias', l.linear.bias)
    print('params', list(l.parameters()))

    """
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(l.parameters(), lr=1e-4)
    for t in range(500):
        y_pred = l(x, X)
        loss = criterion(y_pred, y)
        print(t, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    """
