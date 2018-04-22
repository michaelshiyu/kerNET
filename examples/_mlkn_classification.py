# -*- coding: utf-8 -*-
# torch 0.3.1

from __future__ import division, print_function

import numpy as np
import torch
from torch.autograd import Variable
import torchvision

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
    # large vision datasets
    #########
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=(.5,), std=(1,)),
        torchvision.transforms.ToTensor()
    ])

    root = './torchvision_datasets'
    train = torchvision.datasets.MNIST(
        root=root,
        train=True,
        transform=transform,
        download=True
        )
    test = torchvision.datasets.MNIST(
        root=root,
        train=False,
        transform=transform
        )
    x_train = Variable(
        train.train_data.type(dtype), requires_grad=False
        ).view(60000, -1)
    y_train = Variable(
        train.train_labels.type(dtype), requires_grad=False
        )

    n_class = 10
    """
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=True
    )
    """

    mlkn = MLKNClassifier()
    # create ensemble layers so that large datasets can be fitted into memory
    # note that weight initializations for the layers will be different compared
    # to the ordinary mode
    linear_ensemble0, linear_ensemble1 = kerLinearEnsemble(), kerLinearEnsemble()
    for i, x_train_batch in enumerate(K.get_batch(x_train, batch_size=300)):
        use_bias = True if i==0 else False
        linear_ensemble0.add(
            kerLinear(X=x_train_batch[0], out_dim=300, sigma=5, bias=use_bias)
        )
        linear_ensemble1.add(
            kerLinear(X=x_train_batch[0], out_dim=n_class, sigma=.1, bias=use_bias)
        )
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
        batch_size=100,
        shuffle=True,
        X=x_train,
        Y=y_train,
        n_class=n_class,
        accumulate_grad=False
        )
    # make a prediction on the test set and print error
    y_pred = mlkn.predict(X_test=x_test, X=x_train, batch_size=15)
    err = mlkn.get_error(y_pred, y_test)
    print('error rate: {:.2f}%'.format(err.data[0] * 100))
