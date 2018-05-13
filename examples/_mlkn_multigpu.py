# -*- coding: utf-8 -*-
# torch 0.3.1

from __future__ import division, print_function

import numpy as np
import torch
from torch.autograd import Variable
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, \
load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import sys
sys.path.append('../kernet')
import backend as K
from models.mlkn import MLKN
from layers.kerlinear import kerLinear
from layers.ensemble import kerLinearEnsemble

torch.manual_seed(1234)
np.random.seed(1234)

if __name__=='__main__':
    """
    This example shows how a generic MLKN works. The MLKN implemented here
    inherits only the general architecture from https://arxiv.org/abs/1802.03774
    but not the greedy training method. Thus, it is applicable to any general
    learning problem including classification, regression, etc.
    """
    x, y = load_breast_cancer(return_X_y=True) # 3.51 (acc grad); 2.46
    x, y = load_digits(return_X_y=True) # 10.01 (acc grad); 4.89
    # x, y = load_iris(return_X_y=True) # 1.33 (acc grad); 5.33
    # x, y = load_boston(return_X_y=True) # 0.0263 (acc grad); 0.0275

    task = 'classification' # 'regression' or 'classification'
    ensemble = False
    batch_size=30 # for ensemble layers

    n_gpu = torch.cuda.device_count()
    gpus = list(range(n_gpu))

    # for other Multiple Kernel Learning benchmarks used in the paper, you could
    # do:
    # x = np.load('../kernet/datasets/mkl/name_of_dataset.npy')
    # y = np.load('../kernet/datasets/mkl/name_of_dataset_labels.npy')
    # note that for some of the datasets, the results reported are only on a
    # subset of the data with size given in Table 1. This is to keep consistency
    # with the original paper that reported most of the results.
    # A random subset is chosen at each one of the 20 runs.

    # standardize features to zero-mean and unit-variance
    standard = StandardScaler()
    x = standard.fit_transform(x)

    if task=='regression':
        # 0-1 normalization for y
        y = y.reshape(-1, 1)
        minmax = MinMaxScaler()
        y = minmax.fit_transform(y)


    if task=='regression':
        layer1dim = y.shape[1]
    elif task=='classification':
        layer1dim = int(np.amax(y) + 1)


    dtype = torch.FloatTensor

    X = Variable(torch.from_numpy(x).type(dtype), requires_grad=False)
    Y = Variable(torch.from_numpy(y).type(dtype), requires_grad=False)
    if torch.cuda.is_available():
        X = X.cuda()
        Y = Y.cuda()

    # randomly permute data
    X, Y = K.rand_shuffle(X, Y)

    # split data evenly into training and test
    index = len(X)//2
    x_train, y_train = X[:index], Y[:index]
    x_test, y_test = X[index:], Y[index:]

    mlkn = MLKN()

    layer0 = kerLinear(X=x_train, out_dim=15, sigma=5, bias=True)
    layer1 = kerLinear(X=x_train, out_dim=layer1dim, sigma=.1, bias=True)
    # for non-input layers, pass to X the
    # set of raw data you want to center the kernel machines on,
    # for layer n, layer.X will be updated in runtime to
    # F_n-1(...(F_0(layer.X))...)

    if not ensemble:
        mlkn.add_layer(layer0)
        mlkn.add_layer(layer1)

    else:
        # create ensemble layers so that large datasets can be fitted into memory
        mlkn.add_layer(K.to_ensemble(layer0, batch_size))
        mlkn.add_layer(K.to_ensemble(layer1, batch_size))

    # add optimizer for each layer, this works with any torch.optim.Optimizer
    mlkn.add_optimizer(
        torch.optim.Adam(params=mlkn.parameters(), lr=1e-3, weight_decay=0.1)
        )
    # specify loss function for the output layer, this works with any
    # PyTorch loss function but it is recommended that you use CrossEntropyLoss
    if task=='classification':
        mlkn.add_loss(torch.nn.CrossEntropyLoss())
        mlkn.add_metric(K.L0Loss())
    elif task=='regression':
        mlkn.add_loss(torch.nn.MSELoss())
        mlkn.add_metric(torch.nn.MSELoss())

    if torch.cuda.is_available():
        mlkn.cuda()


    # fit the model
    mlkn.fit(
        n_epoch=30,
        batch_size=30,
        shuffle=True,
        X=x_train,
        Y=y_train,
        accumulate_grad=False,
        X_val=x_train,
        Y_val=y_train,
        val_window=5
        )

    y_raw = mlkn.evaluate(X_test=x_test, Y_test=y_test)

    if task=='regression':
        if torch.cuda.is_available():
            y_raw = y_raw.cpu()
            y_test = y_test.cpu()
        y_raw_np = y_raw.data.numpy()
        y_test_np = y_test.data.numpy()

        y_raw_np = minmax.inverse_transform(y_raw_np)
        y_test_np = minmax.inverse_transform(y_test_np)
        mse = sum((y_raw_np - y_test_np)**2) / len(y_test_np)
        print('MSELoss(original scale): {:.4f}'.format(mse[0]))

    #########
    # using multiple GPUs
    """
    mlkn_parallel = torch.nn.DataParallel(mlkn)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=mlkn_parallel.parameters(), lr=1e-3, weight_decay=0.1
        )
    n_epoch = 30
    batch_size=30
    accumulate_grad=False

    n_batch = x_train.shape[0] // batch_size
    last_batch = bool(x_train.shape[0] % batch_size)

    for _ in range(n_epoch):
        __ = 0
        optimizer.zero_grad()

        for x, y in K.get_batch(x_train, y_train, batch_size=batch_size):
            update_X = True if _==0 and __==0 else False
            __ += 1
            output = mlkn_parallel(x, update_X=update_X)

            loss = loss_fn(output, y)
            print('epoch: {}/{}, batch: {}/{}, loss({}): {:.3f}'.format(
                _+1, n_epoch, __, n_batch+int(last_batch),
                loss_fn.__class__.__name__,
                loss.data[0]
                ))

            if not accumulate_grad:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                mlkn_parallel(x, update_X=True) # update X after each step

            else:
                loss.backward(retain_graph=True)
        if accumulate_grad:
            optimizer.step()
            optimizer.zero_grad()
            mlkn_parallel(x, update_X=True)
    """
