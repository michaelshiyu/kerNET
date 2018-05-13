# -*- coding: utf-8 -*-
# torch 0.3.1

from __future__ import division, print_function

import numpy as np
import torch
from torch.autograd import Variable
import torchvision

from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('../kernet')
import backend as K
from models.mlkn import MLKNClassifier
from layers.kerlinear import kerLinear
from layers.ensemble import kerLinearEnsemble

if __name__=='__main__':
    #########
    # large vision datasets
    #########

    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    '''
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=(.5,), std=(1,)),
        torchvision.transforms.ToTensor()
    ])

    root = './torchvision_datasets'
    train = torchvision.datasets.rectangles_image(
        root=root,
        train=True,
        transform=transform,
        download=True
        )
    test = torchvision.datasets.rectangles_image(
        root=root,
        train=False,
        transform=transform
        )
    x_train = Variable(
        train.train_data.type(dtype), requires_grad=False
        ).view(100000, -1)
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
    '''

    addr = '/Users/michael/Desktop/Github/data/rectangles_image/'
    addr = '/home/michaelshiyu/Github/data/rectangles_image/' # for miner
    # addr = '/home/administrator/Github/data/rectangles_image/' # for lab
    # addr = '/home/paperspace/Github/data/rectangles_image/' # for paperspace
    x_train = Variable(torch.from_numpy(np.load(addr+'rectangles_image_train_img.npy')).type(dtype), requires_grad=False) # when change datasets, change size of validation set
    y_train = Variable(torch.from_numpy(np.load(addr+'rectangles_image_train_label.npy')).type(dtype), requires_grad=False)
    x_test = Variable(torch.from_numpy(np.load(addr+'rectangles_image_test_img.npy')).type(dtype), requires_grad=False)
    y_test = Variable(torch.from_numpy(np.load(addr+'rectangles_image_test_label.npy')).type(dtype), requires_grad=False)
    x_val = x_train[10000:]
    y_val = y_train[10000:]
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    n_class = int(torch.max(y_train) + 1)

    ensemble = True
    batch_size=100

    history = 'rectangles_image1.txt'
    for epo1 in [30, 50, 70]:
        for epo2 in [10, 15, 20]:
            for hidden_dim in [5, 10, 15]:
                for lr1 in [1e-1, 1e-3]:
                    for lr2 in [1e-1, 1e-3]:
                        for w_decay1 in [1e-3, 1e-5, 1e-7]:
                            for w_decay2 in [1e-3, 1e-5, 1e-7]:
                                for sigma1 in [3]:
                                    for sigma2 in [.5]:
                                        for n_center2 in [10000]:
                                            for seed in range(0, 5):

                                                torch.manual_seed(seed)
                                                np.random.seed(seed)

                                                print('sigma1', sigma1, 'sigma2', sigma2, 'epo1', epo1, 'epo2', epo2, 'hidden_dim', hidden_dim, 'lr1', lr1, 'lr2', lr2, 'w_decay1', w_decay1, 'w_decay2', w_decay2, 'n_center2', n_center2, 'seed', seed, file=open(history,'a'))
                                                # print('sigma1', sigma1, 'sigma2', sigma2, 'epo1', epo1, 'epo2', epo2, 'hidden_dim', hidden_dim, 'lr1', lr1, 'lr2', lr2, 'w_decay1', w_decay1, 'w_decay2', w_decay2, 'seed', seed, file=open(history,'a'))

                                                mlkn = MLKNClassifier()

                                                x_train_, y_train_ = K.get_subset(
                                                    X=x_train,
                                                    Y=y_train,
                                                    n=n_center2,
                                                    shuffle=True
                                                    )

                                                layer0 = kerLinear(X=x_train, out_dim=hidden_dim, sigma=sigma1, bias=True)
                                                layer1 = kerLinear(X=x_train_, out_dim=n_class, sigma=sigma2, bias=True)

                                                if not ensemble:
                                                    mlkn.add_layer(layer0)
                                                    mlkn.add_layer(layer1)

                                                else:
                                                    mlkn.add_layer(K.to_ensemble(layer0, batch_size))
                                                    mlkn.add_layer(K.to_ensemble(layer1, batch_size))

                                                mlkn.add_optimizer(
                                                    torch.optim.Adam(params=mlkn.parameters(), lr=lr1, weight_decay=w_decay1)
                                                    )
                                                mlkn.add_optimizer(
                                                    torch.optim.Adam(params=mlkn.parameters(), lr=lr2, weight_decay=w_decay2)
                                                    )

                                                mlkn.add_loss(torch.nn.CrossEntropyLoss())
                                                mlkn.add_metric(K.L0Loss())

                                                if torch.cuda.is_available():
                                                    mlkn.cuda()


                                                mlkn.fit(
                                                    n_epoch=(epo1, epo2),
                                                    batch_size=1000,
                                                    shuffle=True,
                                                    X=x_train,
                                                    Y=y_train,
                                                    n_class=n_class,
                                                    accumulate_grad=False,
                                                    X_val=x_val,
                                                    Y_val=y_val,
                                                    val_window=5,
                                                    write_to=history
                                                    )

                                                # mlkn.evaluate(X_test=x_test, Y_test=y_test, batch_size=300, write_to=history)
