#!/usr/bin/env python

import itertools, sys, os

import numpy as np
import torch
import torchvision

import kernet.backend as K
from kernet.models.feedforward import greedyFeedforward
from kernet.layers.kernelized_layer import kFullyConnected, kFullyConnectedEnsemble

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
    
    n_class = 10
    
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
        x_train3, y_train3 = K.get_subset(
            X=x_train,
            Y=y_train,
            n=n_center3,
            shuffle=True 
            )

        # a kernelized, fully-connected layer. X is the set of centers, n_out is the number of kernel machines on this layer
        layer1 = kFullyConnected(X=x_train, n_out=hidden_dim1, kernel='gaussian', sigma=sigma1, bias=True)
        layer2 = kFullyConnected(X=x_train2, n_out=hidden_dim2, kernel='gaussian', sigma=sigma2, bias=True)
        layer3 = kFullyConnected(X=x_train3, n_out=n_class, kernel='gaussian', sigma=sigma3, bias=True)

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
            net.add_loss(torch.nn.MSELoss(reduction='sum'))
            net.add_loss(torch.nn.MSELoss(reduction='sum'))
            net.add_metric(torch.nn.MSELoss(reduction='sum'))
            net.add_metric(torch.nn.MSELoss(reduction='sum'))
        elif hidden_cost=='l1':
            net.add_loss(torch.nn.L1Loss(reduction='sum'))
            net.add_loss(torch.nn.L1Loss(reduction='sum'))
            net.add_metric(torch.nn.L1Loss(reduction='sum'))
            net.add_metric(torch.nn.L1Loss(reduction='sum'))

        # add loss function for the output layer
        net.add_loss(torch.nn.CrossEntropyLoss(reduction='sum'))
        net.add_metric(K.L0Loss(reduction='sum'))

        # this specifies how the G_i are computed (see the paper for the definition of G_i)
        net.add_critic(layer2.phi) # calculate G_1 using kernel k^(2)
        net.add_critic(layer3.phi)

        #########
        # begin training
        #########
        net._device = device
        net.to(net._device)

        net.fit(
            n_epoch=(epo1, epo2, epo3),
            train_loader=train_loader,
            n_class=n_class,
            accumulate_grad=accumulate_grad,
            # technically we should use a stand-out validation set instead of 
            # the test set, this is just to give you an example of how fit works
            val_loader=test_loader, 
            val_window=1,
            )

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net.state_dict(), './checkpoint/kmlp_mnist.t7')

        #########
        # test
        #########

        net.evaluate(test_loader=test_loader, metric_fn=K.L0Loss(reduction='sum'))

        #########
        # resume from checkpoint
        #########

        # note that pausing and resuming training in a layer-wise setting is somewhat
        # more delicate than they are in backpropagation. For example, consider training 
        # a two-layer model layer-wise, two consecutive training sessions with 
        # epochs (10, 10) and (5, 5) result in a model that is different from that
        # obtained with training for (15, 15). It is the output layer that differs in this 
        # case. Although, you could train the input layer
        # for 10 epochs, pause and then resume for 5 epochs before you start the training
        # of the output layer. And that would give you the same result as if you have 
        # trained the network for (15, 15).

        # also note that if you would like to instantiate a network model from scratch,
        # you should train it for at least 1 epoch before calling net.load_state_dict
        # otherwise torch would most likely throw a size mismatch error because 
        # the tensor of centers in your new model, i.e., yourmodel.X, is not of 
        # the same size as that in the state_dict you're loading
        net.load_state_dict(torch.load('./checkpoint/kmlp_mnist.t7'))
        net.fit(
            n_epoch=(epo1, epo2, epo3),
            train_loader=train_loader,
            n_class=n_class,
            accumulate_grad=accumulate_grad,
            val_loader=test_loader, 
            val_window=1,
            )
        net.evaluate(test_loader=test_loader, metric_fn=K.L0Loss(reduction='sum'))
