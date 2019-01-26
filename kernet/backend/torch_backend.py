#!/usr/bin/env python


import numpy as np
import torch
from torch.nn.modules.loss import _Loss

# TODO tests
# TODO documentation is behind code

class Phi():
    def __init__(self, kernel='gaussian', sigma=1):
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

        x_img : Tensor, shape (batch_size, n_example)
        """
        self.kernel = kernel.lower()

        assert self.kernel in ['gaussian']

        if self.kernel == 'gaussian': 
            self.sigma = sigma
            self.phi = knPhi
            self.a, self.c = 0, 1
        else:
            raise ValueError('currently only supports Gaussian kernel')

    def __call__(self, x, X=None):

        if len(x.shape)==1: x.unsqueeze_(0)
        # TODO check for input with >2 dimensions: (N, n1, n2, ..., H, W)
        # assert len(x.shape)==2
        if len(x.shape)>2:
            x = x.view(x.shape[0], -1)

        if self.kernel == 'gaussian': 
        # Gaussian kernel maps input x against some X: x \mapsto (k(x, x_1), k(x, x_2), ..., k(x, x_N)), where x_1, ..., x_N \in X
            assert X is not None
        
            if len(X.shape)==1: X.unsqueeze_(0)
            if len(X.shape)>2:
                X = X.view(X.shape[0], -1)
               
            assert x.shape[1]==X.shape[1]                
            return self.phi(x, X=X, sigma=self.sigma, kernel='gaussian')

    def get_kmtrx(self, x, y):
    
        if self.kernel=='gaussian': 
        # knPhi can be used to compute kernel matrix directly. See comment in __call__
            return self(x, X=y)

    def get_ideal_kmtrx(self, y1, y2, n_class):
        """
        Ideal kernel matrix.
            k(x_i, x_j) = c if y_i == y_j;
            k(x_i, x_j) = a if y_i != y_j.

        Parameters
        ----------
        y1 : Tensor, shape (n1_example, 1) or (1,) (singleton)
            Categorical labels. Values of categorical labels must be in
            {0, 1, ..., n_class-1}.

        y2 : Tensor, shape (n2_example, 1) or (1,) (singleton)

        n_class : int

        Returns
        -------
        ideal : Tensor, shape (n1_example, n2_example)
        """
        # assert c > a

        y1_onehot, y2_onehot = \
            one_hot(y1, n_class).to(torch.float), one_hot(y2, n_class).to(torch.float)
        # cuda requires arguments of .mm be of float type
        ideal = y1_onehot.mm(y2_onehot.t())

        if self.a!=0:
            a_mask = torch.full_like(ideal, self.a)
            ideal = torch.where(ideal==0, a_mask, ideal)
        
        if self.c!=1:
            c_mask = torch.full_like(ideal, self.c)
            ideal = torch.where(ideal==1, c_mask, ideal)

        return ideal


def knPhi(x, X, sigma, kernel='gaussian'):
    """
    kernel map: map x against X

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
    
    # TODO: check when len(x.shape)>=2
    # TODO: supports only Gaussian kernel for now

    kernel = kernel.lower()
    assert kernel in ['gaussian']

    if kernel=='gaussian':
        x = X.sub(x.unsqueeze(1)).pow_(2).sum(dim=-1).mul_(-1./(2*sigma**2)).exp_()
        
    return x

def categorical(y):
    """
    Convert one-hot labels back to categorical.
    TODO not sure if this method is robust enough

    Parameters
    ----------
    y_onehot : Tensor (n_example, n_class)

    Returns
    -------
    y_cat: (n_example,)
    """
    y_cat = torch.argmax(y, dim=1, keepdim=False)
    return y_cat

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
    # this is because scatter_ only supports LongTensor for its index param

    assert torch.max(y)+1 <= n_class

    y_onehot = y.new_zeros(y.shape[0], n_class)
    ones = y.new_ones(y.shape[0], 1)

    y_onehot.scatter_(1, y, ones)

    return y_onehot

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
    # if batch_size >= lens[0]: yield sets # FIXME

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

    # new_index = torch.randperm(lens[0]) # FIXME: when fit.shuffle=True, this
    # creates different random indices for knFC and knFCEnsemble,
    # which leads to different results
    new_index = np.random.permutation(lens[0])

    sets = list(map(lambda x: x[new_index], sets))

    return sets

def to_ensemble(layer, batch_size):
    """
    Break a layer object into an equivalent ensemble layer object.

    Parameters
    ----------
    layer : knFC
        Supports knFC only.

    Returns
    -------
    ensemble_layer : knFCEnsemble
    """
    from kernet.layers.kn import knFC, knFCEnsemble

    # FIXME throws an error when excuting kn.py, probably some import problem
    # FIXME do not do the import in the beginning, will cause circular
    # inference
    assert isinstance(layer, knFC)

    X = layer.X
    ensemble_layer = knFCEnsemble()
    for i, x in enumerate(get_batch(X, batch_size=batch_size)):
        use_bias = True if layer.bias is not None and i==0 else False
        component = knFC(
            X=x[0],
            n_out=layer.n_out,
            kernel=layer.kernel,
            sigma=layer.phi.sigma, # TODO only support Gaussian kernel
            bias=use_bias
            )

        # FIXME: can it be dangerous to modify params in place?
        component.weight.data = \
                layer.weight[:,i*batch_size:(i+1)*batch_size].clone()

        if use_bias:
            component.bias.data = layer.bias.clone()

        # shallow copy only (create new memory instead of an alias to the
        # original data), this is to prevent the returned ensemble instance share
        # underlying weights with the given layer instance

        ensemble_layer.add_comp(component)
    return ensemble_layer

class L0Loss(_Loss):
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
        # err = (y_pred!=y).sum().type(torch.FloatTensor).div_(float(y.shape[0])) # FIXME throws an error when directly computing using torch objects

        y_pred = y_pred.to('cpu')
        y = y.to('cpu')

        err = float(sum(y_pred.numpy()!=y.numpy())) / y.shape[0]
        # FIXME: for numpy objects converted from torch.tensor, still can access
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
    pass