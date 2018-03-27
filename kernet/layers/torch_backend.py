# -*- coding: utf-8 -*-
# torch 0.3.1

import torch
import math as m

def gaussianKer(x, y, sigma):
    """
    Gaussian kernel: k(x, y) = exp(-||x-y||_2^2 / (2 * sigma^2)).
    Arguments must be matrices or 1darrays. 1darrays will be converted to
    matrices. While the last dimension of its two
    arguments must be identical, this function supports broadcasting in the
    first dimension. Can be used to calculate Gram matrix.

    Parameters
    ----------

    x : Tensor, shape (n1_example, dim)

    y : Tensor, shape (n2_example, dim)

    sigma : scalar

    Returns
    -------

    gram : Tensor, shape (n1_example, n2_example)
        Technically not a Gram matrix when x!=y, only using this name for
        convenience.
    """
    if len(x.shape)==1: x.unsqueeze_(0)
    if len(y.shape)==1: y.unsqueeze_(0)
    assert len(x.shape)==2 and len(y.shape)==2 and x.shape[1]==y.shape[1]
    # TODO: if len(x.shape)>=2 but only two dimensions are nontrivial, should
    # allow computation after squeezing into 2darray
    # TODO: support ndarrays where n>2, e.g., RGB images may have shape
    # (n_example, 3_channels, channel_dim). shape of return should
    # always be (n1_example, n2_example), need to be careful with broadcasting
    # and which dimensions to sum out in this part:
    # y.sub(x.unsqueeze(1)).pow(2).sum(dim=-1)

    gram = y.sub(x.unsqueeze(1)).pow(2).sum(dim=-1).mul(-1./(2*sigma**2)).exp()
    return gram


def kerMap(x, X, sigma):
    """
    For all x_ \in x, computes the image of x_ under the mapping:
        f: f(x_) -> (k(x_1, x_), k(x_2, x_), ..., k(x_n, x_)),
    where k is a kernel function and X = {x_1, x_2, ..., x_n}.
    Currently only supports Gaussian kernel:
    k(x, y) = exp(-||x-y||_2^2 / (2 * sigma^2)).
    Can be used to calculate Gram matrix.

    Parameters
    ----------

    x : Tensor, shape (batch_size, dim)

    X : Tensor, shape (n_example, dim)

    sigma : scalar

    Returns
    -------

    x_image : Tensor, shape (batch_size, n_example)
    """
    x_image = gaussianKer(x, X, sigma)

    return x_image

def one_hot(y, n_class):
    """
    Convert categorical labels to one-hot labels. Values of categorical labels
    must be in {0, 1, ..., n_class-1}. This function performs the most
    straightforward transform: numerical value of y is directly used as the
    positional offset of the 1 in the code, e.g., if y_categorical = 3 and
    n_class = 5, then y_onehot = [0, 0, 0, 1, 0].

    Parameters
    ----------
    y : Tensor, shape (n_example, 1) or (1,) (singleton)

    n_class : int

    Returns
    -------
    y_onehot : Tensor (n_example, n_class)
    """
    assert n_class >= 2
    if len(y.shape)==1: y.unsqueeze_(0)
    assert len(y.shape)==2
    # assert torch.max(y)+1 <= n_class # BUG: bool(Variable) is ambiguous

    y = y.type(torch.LongTensor) # scatter_ requires index to be
    # torch.LongTensor type, if y is of the correct type, this function
    # does nothing and returns the original object

    n_example = y.shape[0]

    y_onehot = torch.FloatTensor(n_example, n_class).fill_(0)
    ones = torch.FloatTensor(n_example, 1).fill_(1)
    y_onehot.scatter_(1, y, ones)
    return y_onehot

def ideal_gram(y1, y2, n_class):
    """
    Ideal Gram matrix for classification.
        k(x_i, x_j) = 1 if y_i == y_j;
        k(x_i, x_j) = 0 if y_i != y_j.

    Parameters
    ----------
    y1 : Tensor, shape (n1_example, 1) or (1,) (singleton)
        Categorical labels. Values of categorical labels must be in
        {0, 1, ..., n_class-1}.

    y2 : Tensor, shape (n2_example, 1) or (1,) (singleton)

    n_class : int

    Returns
    -------
    ideal_gram : Tensor, shape (n1_example, n2_example)
    """
    y1_onehot, y2_onehot = one_hot(y1, n_class), one_hot(y2, n_class)
    ideal_gram = y1_onehot.mm(y2_onehot.transpose(dim0=0, dim1=1))
    return ideal_gram

def frobenius_inner_prod(mat1, mat2):
    """
    Frobenius inner product of two matrices.
    See https://en.wikipedia.org/wiki/Frobenius_inner_product.

    Parameters
    ----------
    mat1, mat2 : Tensor, shape (m, n)

    Returns
    -------
    f : scalar
        Frobenius inner product of mat1 and mat2.
    """
    assert mat1.shape==mat2.shape
    f = mat1.mul(mat2).sum()
    return f


def alignment(gram1, gram2):
    """
    Computes the empirical alignment between two kernels (Gram matrices). See
    http://papers.nips.cc/paper/1946-on-kernel-target-alignment.pdf.

    Parameters
    ----------
    gram1, gram2 : Tensor, shape (m, n)

    Returns
    -------
    alignment : scalar
    """
    # TODO: this is probably not differentiable
    alignment = frobenius_inner_prod(gram1, gram2) /\
        m.sqrt(frobenius_inner_prod(gram1, gram1) *
        frobenius_inner_prod(gram2, gram2))
    return alignment

if __name__=='__main__':
    x = torch.FloatTensor([[1, 2]])
    X = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
    # y = kerMap(x, X, sigma=1)
    y = torch.FloatTensor([[1], [3], [2]])
    y_ = torch.FloatTensor([[2], [1]])
    gram1 = ideal_gram(y, y_, 4)
    print(gram1)
    y = torch.FloatTensor([[3], [3], [3]])
    y_ = torch.FloatTensor([[3], [3]])
    gram2 = ideal_gram(y, y_, 4)
    print(gram2)
    print(frobenius_inner_prod(gram1, gram2))
    print(alignment(gram1, gram2))
