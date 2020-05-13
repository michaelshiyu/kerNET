#!/usr/bin/env python

import itertools, os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import kernet.backend as K
from kernet.models.feedforward import feedforward, greedyFeedforward
from kernet.layers.kernelized_layer import kFullyConnected, kFullyConnectedEnsemble

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

class LeNet5(torch.nn.Module):
    """LeNet5 minus the output layer.
    (N, in_channels, 32, 32) in, (N, 120) out"""

    def __init__(self, in_channels, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=padding)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__=='__main__':
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
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
    
    n_class = 10

    #########
    # set up model
    #########

    params = [
        # model hyperparameters
        [30], # epo1, the number of epochs to train for the first hidden layer 
        [15], # epo2
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

        train_loader = torch.utils.data.DataLoader(
            dataset=train,
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test,
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=2
        )

        net = greedyFeedforward()
        # net = feedforward()

        # randomly get centers for the kernelized layer
        dummy_train_loader = torch.utils.data.DataLoader(
            dataset=train,
            batch_size=60000,
            shuffle=False
        )
        x_train, y_train = next(iter(dummy_train_loader))

        # get a balanced subset of size n as centers
        x_train2, y_train2 = K.get_subset(
            X=x_train,
            Y=y_train,
            n=n_center2,
            shuffle=True 
            )
        
        layer1 = LeNet5_conv(1, padding=2) 
        # layer1 = LeNet5(1, padding=2) 

        # a kernelized, fully-connected layer. X is the set of centers, n_out is the number of kernel machines on this layer
        layer2 = kFullyConnected(X=x_train2, n_out=n_class, kernel='gaussian', sigma=sigma2, bias=True)
        
        if ensemble:
            layer2 = layer2.to_ensemble(component_size)
        
        # add optimizer to each layer. There is no need to assign each optimizer to the parameters of the corresponding layer manually, this will later be done by the model in net._compile() when you call net.fit(). 
        net.add_optimizer(
            torch.optim.Adam(params=layer1.parameters(), lr=lr1, weight_decay=w_decay1) 
            )
        net.add_optimizer(
            torch.optim.Adam(params=layer2.parameters(), lr=lr2, weight_decay=w_decay2)
            )
        
        # add loss function for the hidden layers
        # for all loss functions except CosineSimilarity, set reduction to 'sum'
        # and kerNET would do the averagings for you and return loss values as if 
        # the loss functions have been set to reduction='mean'
        
        if hidden_cost=='alignment': # changing between alignment, l1 and l2 may require re-tuning of the hyperparameters
            net.add_loss(torch.nn.CosineSimilarity())
            net.add_metric(torch.nn.CosineSimilarity()) # metric for validation
        elif hidden_cost=='l2':
            net.add_loss(torch.nn.MSELoss(reduction='sum'))
            net.add_metric(torch.nn.MSELoss(reduction='sum'))
        elif hidden_cost=='l1':
            net.add_loss(torch.nn.L1Loss(reduction='sum'))
            net.add_metric(torch.nn.L1Loss(reduction='sum'))
        
        # add loss function for the output layer
        net.add_loss(torch.nn.CrossEntropyLoss(reduction='sum'))
        net.add_metric(K.L0Loss(reduction='sum'))

        # this specifies how the G_i are computed (see the paper for the definition of G_i)
        net.add_critic(layer2.phi) # calculate G_1 using kernel k^(2)
        """
        if torch.cuda.device_count() > 1:
            layer1 = torch.nn.DataParallel(layer1)
            # FIXME parallelizing kernelized layers currently has issues with layer.X
            # layer2 = torch.nn.DataParallel(layer2)
        """
        net.add_layer(layer1)
        net.add_layer(layer2)

        #########
        # begin training
        #########
        net._device = device
        net.to(net._device)

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        net.fit(
            n_epoch=(epo1, epo2),
            # n_epoch=30,
            train_loader=train_loader,
            n_class=n_class,
            accumulate_grad=accumulate_grad,
            # technically we should use a stand-out validation set instead of 
            # the test set, this is just to give you an example of how fit works
            val_loader=test_loader, 
            val_window=1,
            save_best=True,
            logdir='./checkpoint/test_logger_layerwise.t7'
            )
        
        net.evaluate(test_loader=test_loader, metric_fn=K.L0Loss(reduction='sum'))
        
        log = torch.load('checkpoint/test_logger_layerwise.t7')
        print(log['layer0_log'].keys(), log['layer1_log'].keys(), log.keys())
        net.load_state_dict(log['model_state_dict'])

        #########
        # test
        #########

        net.evaluate(test_loader=test_loader, metric_fn=K.L0Loss(reduction='sum'))
