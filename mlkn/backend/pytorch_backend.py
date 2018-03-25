import torch
import numpy as np

def kerMap(x, X, sigma):
    """
    Computes the image of x under the mapping:
        f: f(x) -> (k(x_1, x), k(x_2, x), ..., k(x_n, x)),
    where k is a kernel function. Currently only supports Gaussian kernel:
    k(x, y) = exp(-||x-y||_2^2 / (2 * sigma^2)).

    Parameters
    ----------

    x : Tensor, shape (batch_size, dim)

    X : Tensor, shape (n_example, dim)

    sigma : scalar

    Returns
    -------

    x_image : Tensor, shape (batch_size, n_example)
    """
    if len(x.shape)==1: x.unsqueeze_(0)
    if len(X.shape)==1: X.unsqueeze_(0)
    assert x.shape[0]<=X.shape[0] and x.shape[1]==X.shape[1]

    x_image = x.sub(X).pow(2).sum(dim=1).mul(-1./(2*sigma**2)).exp()

    print(x_image)

if __name__=='__main__':
    x = torch.FloatTensor([[1, 2], [3, 4]])
    X = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
    kerMap(x, X, 1)
