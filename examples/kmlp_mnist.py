#!/usr/bin/env python

import itertools, sys

import numpy as np
import torch
import torchvision

import kernet.backend as K
from kernet.models.ffc import greedyFFC
from kernet.layers.kn import knFC, knFCEnsemble

"""Training a kMLP layer-wise for MNIST. This setting should give an error rate of about 1.5%."""

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #########
    # load data
    #########

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
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
        transform=transform,
        download=True
        )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train,
        batch_size=60000,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test,
        batch_size=60000,
        shuffle=False
    )

    x_train, y_train = iter(train_loader).next()
    x_test, y_test = iter(test_loader).next()

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)
    
    n_class = int(torch.max(y_train) + 1)
    
    #########
    # set up model
    #########

    params = [
        # model hyperparameters
        [25], # epo1, the number of epochs to train for the first hidden layer 
        [15], # epo2 
        [5], # epo3
        [100], # hidden_dim1
        [100], # hidden_dim2
        [5e-2], # lr1
        [5e-4], # lr2
        [1e-4], # lr3
        [1e-7/4.5], # w_decay1,
        [1e-5/4.5], # w_decay2,
        [1e-5/4.5], # w_decay3,
        [3.5], # sigma1, kernel width of the Gaussian kernels on the first hidden layer
        [11], # sigma2
        [11], # sigma3
        [60000], # n_center2, the number of centers to randomly retain for the kernel machines on the second hidden layer, this implements the acceleration trick proposed in the paper
        [10000], # n_center3

        # memory-saving settings
        [True], # whether to break each kernelized layer into a bunch of smaller layers to save memory, changing this setting does not affect performance
        [300], # component_size, size of each small layer

        # training settings
        [600], # batch_sie
        [True], # shuffle
        [False], # accumulate_grad, whether to accumulate gradient from minibatches and update only at the end of each epoch
        ['alignment'], # hidden_cost,
        ]

    for epo1, epo2, epo3, hidden_dim1, hidden_dim2, lr1, lr2, lr3, w_decay1, \
    w_decay2, w_decay3, sigma1, sigma2, sigma3, n_center2, n_center3, \
    ensemble, component_size, batch_size, shuffle, accumulate_grad, hidden_cost \
    in itertools.product(*params):

        net = greedyFFC()

        # randomly get centers for the kernelized layers
        x_train2, y_train2 = K.get_subset(
            X=x_train,
            Y=y_train,
            n=n_center2,
            shuffle=True 
            )
        x_train3, y_train3 = K.get_subset(
            X=x_train,
            Y=y_train,
            n=n_center3,
            shuffle=True 
            )

        # a kernelized, fully-connected layer. X is the set of centers, n_out is the number of kernel machines on this layer
        layer1 = knFC(X=x_train, n_out=hidden_dim1, kernel='gaussian', sigma=sigma1, bias=True)
        layer2 = knFC(X=x_train2, n_out=hidden_dim2, kernel='gaussian', sigma=sigma2, bias=True)
        layer3 = knFC(X=x_train3, n_out=n_class, kernel='gaussian', sigma=sigma3, bias=True)

        if not ensemble:
            net.add_layer(layer1)
            net.add_layer(layer2)
            net.add_layer(layer3)

        else:
            net.add_layer(layer1.to_ensemble(component_size))
            net.add_layer(layer2.to_ensemble(component_size))
            net.add_layer(layer3.to_ensemble(component_size))

        # add optimizer to each layer. There is no need to assign each optimizer to the parameters of the corresponding layer manually, this will later be done by the model in net._compile() when you call net.fit(). 
        net.add_optimizer(
            torch.optim.Adam(params=net.parameters(), lr=lr1, weight_decay=w_decay1)
            )
        net.add_optimizer(
            torch.optim.Adam(params=net.parameters(), lr=lr2, weight_decay=w_decay2)
            )
        net.add_optimizer(
            torch.optim.Adam(params=net.parameters(), lr=lr3, weight_decay=w_decay3)
            )

        # add loss function for the hidden layers
        if hidden_cost=='alignment':# changing between alignment, l1 and l2 may require re-tuning of the hyperparameters
            net.add_loss(torch.nn.CosineSimilarity())
            net.add_loss(torch.nn.CosineSimilarity())
            net.add_metric(torch.nn.CosineSimilarity()) # metric for validation
            net.add_metric(torch.nn.CosineSimilarity())
        elif hidden_cost=='l2': 
            net.add_loss(torch.nn.MSELoss(size_average=True, reduce=True))
            net.add_loss(torch.nn.MSELoss(size_average=True, reduce=True))
            net.add_metric(torch.nn.MSELoss(size_average=True, reduce=True))
            net.add_metric(torch.nn.MSELoss(size_average=True, reduce=True))
        elif hidden_cost=='l1':
            net.add_loss(torch.nn.L1Loss(size_average=True, reduce=True))
            net.add_loss(torch.nn.L1Loss(size_average=True, reduce=True))
            net.add_metric(torch.nn.L1Loss(size_average=True, reduce=True))
            net.add_metric(torch.nn.L1Loss(size_average=True, reduce=True))

        # add loss function for the output layer
        net.add_loss(torch.nn.CrossEntropyLoss())
        net.add_metric(K.L0Loss())

        # this specifies how the G_i are computed (see the paper for the definition of G_i)
        net.add_critic(layer2.phi) # calculate G_1 using kernel k^(2)
        net.add_critic(layer3.phi)

        #########
        # begin training
        #########

        net.to(device)
        net.fit(
            n_epoch=(epo1, epo2, epo3),
            batch_size=batch_size,
            shuffle=shuffle,
            X=x_train,
            Y=y_train,
            n_class=n_class,
            accumulate_grad=accumulate_grad,
            )

        #########
        # test
        #########

        net.evaluate(X_test=x_test, Y_test=y_test, batch_size=1000, metric_fn=K.L0Loss())
