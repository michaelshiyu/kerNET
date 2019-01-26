#!/usr/bin/env python

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import kernet.backend as K
from kernet.models.ffc import greedyFFC
from kernet.layers.kn import knFC, knFCEnsemble

"""Kernelize the output layer of LeNet-5 and train it layer-wise. This setting should give an error rate of about 0.8%."""

class LeNet5_conv(torch.nn.Module):
    """LeNet5 minus the output layer.
    (N, in_channels, 32, 32) in, (N, 120) out"""

    def __init__(self, in_channels, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=padding)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
    
    
    x_train = x_train.reshape(x_train.shape[0], -1).view(60000, 1, 28, 28) 
    x_test = x_test.reshape(x_test.shape[0], -1).view(10000, 1, 28, 28) 

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)
    
    n_class = int(torch.max(y_train) + 1)

    #########
    # set up model
    #########

    params = [
        # model hyperparameters
        [35], # epo1, the number of epochs to train for the first hidden layer 
        [10], # epo2
        [5e-4], # lr1
        [5e-4], # lr2
        [1e-6], # w_decay1,
        [1e-6], # w_decay2,
        [9], # sigma2, kernel width of the Gaussian kernels on the first hidden layer
        [10000], # n_center2, the number of centers to randomly retain for the kernel machines on the second hidden layer, this implements the acceleration trick proposed in the paper

        # memory-saving settings
        [True], # whether to break each kernelized layer into a bunch of smaller layers to save memory, changing this setting does not affect performance
        [300], # component_size, size of each small layer

        # training settings
        [600], # batch_sie
        [True], # shuffle
        [False], # accumulate_grad, whether to accumulate gradient from minibatches and update only at the end of each epoch
        ['alignment'], # hidden_cost,
        ]

    for epo1, epo2, lr1, lr2, w_decay1, \
    w_decay2, sigma2, n_center2, \
    ensemble, component_size, batch_size, shuffle, accumulate_grad, hidden_cost \
    in itertools.product(*params):

        net = greedyFFC()

        # randomly get centers for the kernelized layer
        x_train2, y_train2 = K.get_subset(
            X=x_train,
            Y=y_train,
            n=n_center2,
            shuffle=True 
            )

        layer1 = LeNet5_conv(1, padding=2) 

        # a kernelized, fully-connected layer. X is the set of centers, n_out is the number of kernel machines on this layer
        layer2 = knFC(X=x_train2, n_out=n_class, kernel='gaussian', sigma=sigma2, bias=True)

        if not ensemble:
            net.add_layer(layer1)
            net.add_layer(layer2)

        else:
            net.add_layer(layer1)
            net.add_layer(layer2.to_ensemble(component_size))

        # add optimizer to each layer. There is no need to assign each optimizer to the parameters of the corresponding layer manually, this will later be done by the model in net._compile() when you call net.fit(). 
        net.add_optimizer(
            torch.optim.Adam(params=net.parameters(), lr=lr1, weight_decay=w_decay1) 
            )
        net.add_optimizer(
            torch.optim.Adam(params=net.parameters(), lr=lr2, weight_decay=w_decay2)
            )

        # add loss function for the hidden layers
        if hidden_cost=='alignment': # changing between alignment, l1 and l2 may require re-tuning of the hyperparameters
            net.add_loss(torch.nn.CosineSimilarity())
            net.add_metric(torch.nn.CosineSimilarity()) # metric for validation
        elif hidden_cost=='l2':
            net.add_loss(torch.nn.MSELoss(size_average=True, reduce=True))
            net.add_metric(torch.nn.MSELoss(size_average=True, reduce=True))
        elif hidden_cost=='l1':
            net.add_loss(torch.nn.L1Loss(size_average=True, reduce=True))
            net.add_metric(torch.nn.L1Loss(size_average=True, reduce=True))

        # add loss function for the output layer
        net.add_loss(torch.nn.CrossEntropyLoss())
        net.add_metric(K.L0Loss())

        # this specifies how the G_i are computed (see the paper for the definition of G_i)
        net.add_critic(layer2.phi) # calculate G_1 using kernel k^(2)

        #########
        # begin training
        #########

        net.to(device)
        net.fit(
            n_epoch=(epo1, epo2),
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
