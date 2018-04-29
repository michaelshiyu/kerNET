# -*- coding: utf-8 -*-
# torch 0.3.1

from __future__ import division, print_function

import numpy as np
import torch
from torch.autograd import Variable
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('../kernet')
import backend as K
from models.mlkn import MLKNClassifier
from layers.kerlinear import kerLinear
from layers.ensemble import kerLinearEnsemble

torch.manual_seed(1234)

if __name__=='__main__':
    """
    This example demonstrates how a MLKN classifier works. Everything in here
    including the architecture of the learning machine and the training
    algorithm strictly follows this paper: https://arxiv.org/abs/1802.03774.
    """
    #########
    # MKL benchmarks
    #########
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor

    x, y = load_breast_cancer(return_X_y=True) # ens 2.46; 2.81 (acc grad)/ ens 3.51; 2.11
    # x, y = load_digits(return_X_y=True) # ens 2.46; 4.34 (acc grad)/ ens 4.78; 5.23
    # x, y = load_iris(return_X_y=True) # ens 4.00; 4.00 (acc grad)/ ens 4.00; 4.00

    ensemble = False
    batch_size=30

    # for other Multiple Kernel Learning benchmarks used in the paper, you could
    # do:
    # x = np.load('../kernet/datasets/mkl/name_of_dataset.npy')
    # y = np.load('../kernet/datasets/mkl/name_of_dataset_labels.npy')
    # note that for some of the datasets, the results reported are only on a
    # subset of the data with size given in Table 1. This is to keep consistency
    # with the original paper that reported most of the results.
    # A random subset is chosen at each one of the 20 runs.

    # standardize features to zero-mean and unit-variance
    normalizer = StandardScaler()
    x = normalizer.fit_transform(x)
    n_class = int(np.amax(y) + 1)

    X = Variable(torch.from_numpy(x).type(dtype), requires_grad=False)
    Y = Variable(torch.from_numpy(y).type(dtype), requires_grad=False)

    # randomly permute data
    X, Y = K.rand_shuffle(X, Y)

    # split data evenly into training and test
    index = len(X)//2
    x_train, y_train = X[:index], Y[:index]
    x_test, y_test = X[index:], Y[index:]

    mlkn = MLKNClassifier()

    layer0 = kerLinear(X=x_train, out_dim=15, sigma=5, bias=True)
    layer1 = kerLinear(X=x_train, out_dim=n_class, sigma=.1, bias=True)
    # for non-input layers, pass to X the
    # set of raw data you want to center the kernel machines on,
    # for layer n, layer.X will be updated in runtime to
    # F_n-1(...(F_0(layer.X))...)

    if not ensemble:
        # add layers to the model, see layers/kerlinear for details on kerLinear
        mlkn.add_layer(layer0)
        mlkn.add_layer(layer1)

    else:
        # create ensemble layers so that large datasets can be fitted into memory
        # note that weight initializations for the layers will be different compared
        # to the ordinary mode

        linear_ensemble0, linear_ensemble1 = kerLinearEnsemble(), kerLinearEnsemble()

        for i, x_train_batch in enumerate(
            K.get_batch(x_train, batch_size=batch_size)
            ):

            use_bias = True if i==0 else False
            component0 = kerLinear(
                X=x_train_batch[0],
                out_dim=15,
                sigma=5,
                bias=use_bias
                )

            component0.weight.data = \
                layer0.weight[:,i*batch_size:(i+1)*batch_size].data
            if use_bias:
                component0.bias.data = layer0.bias.data

            component1 = kerLinear(
                X=x_train_batch[0],
                out_dim=n_class,
                sigma=.1,
                bias=use_bias
                )

            component1.weight.data = \
                layer1.weight[:,i*batch_size:(i+1)*batch_size].data
            if use_bias:
                component1.bias.data = layer1.bias.data

            linear_ensemble0.add(component0)
            linear_ensemble1.add(component1)

        mlkn.add_layer(linear_ensemble0)
        mlkn.add_layer(linear_ensemble1)

    # add optimizer for each layer, this works with any torch.optim.Optimizer
    # note that this model is trained with the proposed layerwise training
    # method by default
    mlkn.add_optimizer(
        torch.optim.Adam(params=mlkn.parameters(), lr=1e-3, weight_decay=0.1)
        )
    mlkn.add_optimizer(
        torch.optim.Adam(params=mlkn.parameters(), lr=1e-3, weight_decay=.1)
        )
    # specify loss function for the output layer, this works with any
    # PyTorch loss function but it is recommended that you use CrossEntropyLoss
    mlkn.add_loss(torch.nn.CrossEntropyLoss())
    if torch.cuda.is_available():
        mlkn.cuda()
    # fit the model
    mlkn.fit(
        n_epoch=(30, 30),
        batch_size=30,
        shuffle=True,
        X=x_train,
        Y=y_train,
        n_class=n_class,
        accumulate_grad=True
        )
    """
    for p in mlkn.layer0.parameters():
        print(p)
    for p in mlkn.layer1.parameters():
        print(p)
    """

    # make a prediction on the test set and print error
    y_pred = mlkn.predict(X_test=x_test, batch_size=15)
    err = mlkn.get_error(y_pred, y_test)
    print('error rate: {:.2f}%'.format(err.data[0] * 100))
