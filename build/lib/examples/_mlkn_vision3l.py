# -*- coding: utf-8 -*-
# torch 0.4.0

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=(.5,), std=(1,)),
        torchvision.transforms.ToTensor()
    ])

    root = './torchvision_datasets'
    train = torchvision.datasets.rectangles(
        root=root,
        train=True,
        transform=transform,
        download=True
        )
    test = torchvision.datasets.rectangles(
        root=root,
        train=False,
        transform=transform
        )
    x_train = Variable(
        train.train_data.type(dtype), requires_grad=False
        ).view(1000, -1)
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

    addr = '/Users/michael/Desktop/Github/data/rectangles/'
    addr = '/home/michaelshiyu/Github/data/rectangles/' # for miner
    # addr = '/home/administrator/Github/data/rectangles/' # for lab
    # addr = '/home/paperspace/Github/data/rectangles/' # for paperspace
    x_train = torch.tensor(np.load(addr+'rectangles_train_img.npy'), dtype=torch.float, device=device, requires_grad=False) # when change datasets, change size of validation set
    y_train = torch.tensor(np.load(addr+'rectangles_train_label.npy'), dtype=torch.int64, device=device, requires_grad=False)
    x_test = torch.tensor(np.load(addr+'rectangles_test_img.npy'), dtype=torch.float, device=device, requires_grad=False)
    y_test = torch.tensor(np.load(addr+'rectangles_test_label.npy'), dtype=torch.int64, device=device, requires_grad=False)
    x_val = x_train[1000:]
    y_val = y_train[1000:]
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    n_class = int(torch.max(y_train) + 1)

    ensemble = True
    batch_size=100

    history = 'rectangles3l.txt'
    # history = None
    for epo0 in [10, 20, 30]:
        for epo1 in [10, 20, 30]:
            for epo2 in [10, 20, 30]:
                for hidden_dim0 in [5, 10, 15]:
                    for hidden_dim1 in [5, 10, 15]:
                        for lr0 in [1e-1, 1e-2, 1e-3]:
                            for lr1 in [1e-1, 1e-2, 1e-3]:
                                for lr2 in [1e-1, 1e-2, 1e-3]:
                                    for w_decay0 in [1e-1, 1e-3, 1e-5]:
                                        for w_decay1 in [1e-1, 1e-3, 1e-5]:
                                            for w_decay2 in [1e-1, 1e-3, 1e-5]:
                                                for sigma0 in [1, 3, 5, 7, 9]:
                                                    for sigma1 in [1, 2, 3, 4, 5]:
                                                        for sigma2 in [.1, .5, 1, 1.5, 2, 2.5]:
                                                            for n_center2 in [100, 500, 1000]:
                                                                for seed in range(0, 3):

                                                                    torch.manual_seed(seed)
                                                                    np.random.seed(seed)

                                                                    if history: print(
                                                                        'sigma0', sigma0,
                                                                        'sigma1', sigma1,
                                                                        'sigma2', sigma2,
                                                                        'epo0', epo0,
                                                                        'epo1', epo1,
                                                                        'epo2', epo2,
                                                                        'hidden_dim0', hidden_dim0,
                                                                        'hidden_dim1', hidden_dim1,
                                                                        'lr0', lr0,
                                                                        'lr1', lr1,
                                                                        'lr2', lr2,
                                                                        'w_decay0', w_decay0,
                                                                        'w_decay1', w_decay1,
                                                                        'w_decay2', w_decay2,
                                                                        'n_center2', n_center2,
                                                                        'seed', seed,
                                                                        file=open(history,'a')
                                                                        )

                                                                    mlkn = MLKNClassifier()

                                                                    x_train_, y_train_ = K.get_subset(
                                                                        X=x_train,
                                                                        Y=y_train,
                                                                        n=n_center2,
                                                                        shuffle=True
                                                                        )

                                                                    layer0 = kerLinear(X=x_train, out_dim=hidden_dim0, sigma=sigma0, bias=True)
                                                                    layer1 = kerLinear(X=x_train, out_dim=hidden_dim1, sigma=sigma1, bias=True)
                                                                    layer2 = kerLinear(X=x_train_, out_dim=n_class, sigma=sigma2, bias=True)

                                                                    if not ensemble:
                                                                        mlkn.add_layer(layer0)
                                                                        mlkn.add_layer(layer1)
                                                                        mlkn.add_layer(layer2)


                                                                    else:
                                                                        mlkn.add_layer(K.to_ensemble(layer0, batch_size))
                                                                        mlkn.add_layer(K.to_ensemble(layer1, batch_size))
                                                                        mlkn.add_layer(K.to_ensemble(layer2, batch_size))

                                                                    mlkn.add_optimizer(
                                                                        torch.optim.Adam(params=mlkn.parameters(), lr=lr0, weight_decay=w_decay0)
                                                                        )
                                                                    mlkn.add_optimizer(
                                                                        torch.optim.Adam(params=mlkn.parameters(), lr=lr1, weight_decay=w_decay1)
                                                                        )
                                                                    mlkn.add_optimizer(
                                                                        torch.optim.Adam(params=mlkn.parameters(), lr=lr2, weight_decay=w_decay2)
                                                                        )

                                                                    mlkn.add_loss(torch.nn.CrossEntropyLoss())
                                                                    mlkn.add_metric(K.L0Loss())

                                                                    mlkn.to(device)


                                                                    mlkn.fit(
                                                                        n_epoch=(epo0, epo1, epo2),
                                                                        batch_size=300,
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

                                                                    mlkn.evaluate(X_test=x_test, Y_test=y_test, batch_size=5000, write_to=history)
