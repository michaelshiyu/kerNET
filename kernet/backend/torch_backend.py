# -*- coding: utf-8 -*-
# torch 0.4.0

from __future__ import division, print_function

import math as m
import numpy as np
import torch
from torch.autograd import Variable

from kernet.layers.kerlinear import kerLinear
from kernet.layers.ensemble import kerLinearEnsemble

# TODO: tests

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
    # TODO: for ndarrays where n>2, e.g., RGB images may have shape
    # (n_example, channels, height, width), stretch into a long vector
    # (n_example, channels*height*width) before
    # passing to this function. this does not make any difference to result.

    gram = y.sub(x.unsqueeze(1)).pow_(2).sum(dim=-1).mul_(-1./(2*sigma**2)).exp_()
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
    # if x or X is not a Variable, must be the buffer self.X or self.X_init,
    # hence there is no need to require grad for them
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
        dtype of y_onehot is consistent with that of y.
    """
    # NOTE: this function is not differentiable
    assert n_class >= 2
    if len(y.shape)==1: y.unsqueeze_(0)
    assert len(y.shape)==2

    y = y.type(torch.int64)
    # NOTE: this is because scatter_ only supports LongTensor for its index param

    assert torch.max(y)+1 <= n_class

    y_onehot = y.new_zeros(y.shape[0], n_class)
    ones = y.new_ones(y.shape[0], 1)

    y_onehot.scatter_(1, y, ones)

    return y_onehot

def ideal_gram(y1, y2, n_class, lower_lim=0):
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

    lower_lim (optional) : float
        Value for k(x_i, x_j) when y_i != y_j

    Returns
    -------
    ideal_gram : Tensor, shape (n1_example, n2_example)
    """
    # TODO: ideal gram for general kernels (lower_lim may no longer be 0 by
    # default, should be min_{x, y\in X} k(x, y); k(x_i, x_i) may no longer be
    # 1)
    y1_onehot, y2_onehot = \
        one_hot(y1, n_class).to(torch.float), one_hot(y2, n_class).to(torch.float)
    # cuda requires arguments of .mm be of float type
    ideal_gram = y1_onehot.mm(y2_onehot.transpose(dim0=0, dim1=1))

    if lower_lim!=0:
        lower_lim_mask = torch.full_like(ideal_gram, lower_lim)
        ideal_gram = torch.where(ideal_gram==0, lower_lim_mask, ideal_gram)
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
    # assert isinstance(mat1, Variable) and isinstance(mat2, Variable))
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
    # BUG: this loss function causes abnormal optimization behaviors, see
    # comments in past commits

    alignment = frobenius_inner_prod(gram1, gram2) /\
        m.sqrt(frobenius_inner_prod(gram1, gram1) *
        frobenius_inner_prod(gram2, gram2))
    return alignment

def get_batch(*sets, batch_size, shuffle=False):
    """
    Generator, break a random sample X into batches of size batch_size.
    The last batch may be of a smaller size. If shuffle, X is shuffled
    before getting the batches.

    This light-weight function is to be used for small, simple datasets.
    For large datasets and better management of memory and multiprocessing,
    consider wrap the data into a torch.utils.data.Dataset object and
    use torch.utils.data.DataLoader.

    Whenever the argument sets has only 1 set, add [0] to the return value.

    Parameters
    ----------

    X1 : Tensor, shape (n_example, dim_1, ..., dim_d1)

    X2 : Tensor, shape (n_example, dim_1, ..., dim_d2)

    ...

    batch_size : int

    shuffle (optional) : bool

    Returns
    -------
    x1 : Tensor, shape (batch_size, dim_1, ..., dim_d1)

    x2 : Tensor, shape (batch_size, dim_1, ..., dim_d2)

    ...
    """
    assert batch_size > 0

    lens = list(map(lambda x: x.shape[0], sets))
    assert lens.count(lens[0])==len(lens) # make sure all sets are equal in
    # sizes of their 1st dims
    if shuffle: sets = rand_shuffle(sets)

    if batch_size >= lens[0]: batch_size = lens[0]
    # if batch_size >= lens[0]: yield sets # BUG

    n_batch = lens[0] // batch_size
    last_batch = bool(lens[0] % batch_size)

    for i in range(n_batch):
        yield tuple(map(lambda x: x[i*batch_size: (i+1)*batch_size], sets))
    if last_batch:
        i += 1
        yield tuple(map(lambda x: x[i*batch_size:], sets))

def rand_shuffle(*sets):
    """
    Shuffle the given sets along the first dimension.

    Parameters
    ----------
    X1 : Tensor, shape (n_example, dim_1, ..., dim_d1)

    X2 : Tensor, shape (n_example, dim_1, ..., dim_d2)

    ...

    Returns
    -------
    X1 : Tensor, shape (n_example, dim_1, ..., dim_d1)
        Shuffled X1.
    X2 : Tensor, shape (n_example, dim_1, ..., dim_d2)

    ...
    """
    # TODO: check if the caller passed in a tuple of sets, this happens when
    # rand_shuffle is called by get_batch. there must be a more elegant way of
    # doing this. and this is not immume to cases where there are multiple
    # nested tuples of 1 element, i.e., things like (((a, b),),)
    if len(sets)==1 and isinstance(sets[0], tuple): sets = sets[0]

    lens = list(map(lambda x: x.shape[0], sets))
    assert lens.count(lens[0])==len(lens) # make sure all sets are equal in
    # sizes of their 1st dims

    # new_index = torch.randperm(lens[0]) # BUG: when fit.shuffle=True, this
    # creates different random indices for kerLinear and kerLinearEnsemble,
    # which leads to different results
    new_index = np.random.permutation(lens[0])

    sets = list(map(lambda x: x[new_index], sets))

    return sets

def to_ensemble(layer, batch_size):
    """
    Break a layer object into an equivalent ensemble layer object.

    Parameters
    ----------
    layer : kerLinear
        Supports kerLinear only.

    Returns
    -------
    ensemble_layer : kerLinearEnsemble
    """
    assert isinstance(layer, kerLinear)
    X = layer.X
    ensemble_layer = kerLinearEnsemble()
    for i, x in enumerate(get_batch(X, batch_size=batch_size)):

        use_bias = True if i==0 else False
        component = kerLinear(
            X=x[0],
            out_dim=layer.out_dim,
            sigma=layer.sigma,
            bias=use_bias
            )

        # BUG: can it be dangerous to modify params in place?
        component.weight.data = \
                layer.weight[:,i*batch_size:(i+1)*batch_size].clone()

        if use_bias:
            component.bias.data = layer.bias.clone()

        # shallow copy only (create new memory instead of an alias to the
        # original data), this is to prevent the returned ensemble instance share
        # underlying weights with the given layer instance

        ensemble_layer.add(component)
    return ensemble_layer

class L0Loss:
    def __call__(self, y_pred, y):
        """
        Compute prediction error rate.

        Parameters
        ----------

        y_pred : Tensor, shape (batch_size,)
            Predicted labels.

        y : Tensor, shape (batch_size,)
            True labels.

        Returns
        -------
        err : scalar
            Error rate.
        """
        assert y_pred.shape==y.shape

        # y_pred = y_pred.type_as(y)
        # err = (y_pred!=y).sum().type(torch.FloatTensor).div_(float(y.shape[0])) # BUG

        y_pred = y_pred.to('cpu')
        y = y.to('cpu')

        err = float(sum(y_pred.numpy()!=y.numpy())) / y.shape[0]
        # BUG: for numpy objects converted from torch.tensor, still can access
        # x.data but x.data is some memory location
        return err

def get_subset(X, Y, n, shuffle=True):
    """
    Get a balanced subset from the given set, i.e., subset with an
    equal number of examples from each class.

    Parameters
    ----------
    X : Tensor (n_example, n_dim)

    Y : Tensor (n_example,) or (n_example, 1)
        Categorical class labels. assert set(Y)==(0, 1, ..., n_class-1)

    n : int
        Number of total examples to collect.

    shuffle (optional) : bool

    Returns
    -------
    x : Tensor (n, n_dim)

    y : (n,) or (n, 1)
    """
    if n > X.shape[0]: n = X.shape[0]
    assert X.shape[0]==Y.shape[0]

    X_ = X.to('cpu').numpy()
    Y_ = Y.to('cpu').numpy()

    n_class = int(max(Y_) + 1)
    indices = {}

    for i in range(n_class):
        indices[i] = np.where(Y_==i)[0]

    batch = n // n_class
    leftover = n % n_class
    # each class should have at least as many as 'batch' examples
    x_ = np.concatenate([X_[indices[i][:batch]] for i in range(n_class)], axis=0)
    y_ = np.concatenate([Y_[indices[i][:batch]] for i in range(n_class)], axis=0)

    i = 0
    while i < leftover:
        x_ = np.concatenate((
            x_,
            X_[indices[i % n_class][batch + 1 + i // n_class]][np.newaxis,]
            ))
        y_ = np.concatenate((
            y_,
            Y_[indices[i % n_class][batch + 1 + i // n_class]][np.newaxis,]
            ))
        i += 1

    x = torch.tensor(x_, dtype=X.dtype, device=X.device, requires_grad=False)
    y = torch.tensor(y_, dtype=Y.dtype, device=Y.device, requires_grad=False)

    if shuffle:
        x, y = rand_shuffle(x, y)

    return x, y


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
