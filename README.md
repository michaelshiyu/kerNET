# kerNET

**(Jan. 27, 2019) This repo is under active development. I am trying to make it easier to use and more memory efficient. Feel free to open an issue if you find something that doesn't work as expected. Also, I should remind you that some documentations are behind the actual code. I'm still working on that.**

kerNET is a [Keras](https://keras.io/)-like wrapper for PyTorch that makes it easier to build kernel networks and a layer-wise learning algorithm proposed in this paper (TDOO: url to be added).

Dependencies:
- Python 3.6
- PyTorch 1.0
- NumPy 1.15

To install, clone this repo to your local machine, go to the directory where the files live and do  ```python setup.py install```. 

Hope you enjoy this repo and any suggestion or contribution would be greatly appreciated!

Some simple tutorials are given below (they assume that you have some basic knowledge about PyTorch):

---------

## Build an RBF network for classification

```python
import torch

import kernet.backend as K
from kernet.models.feedforward import feedforward
from kernet.layers.kernelized_layer import kFullyConnected

# suppose we already have x_train, y_train, x_validation, y_validation, x_test, 
# y_test and n_class (the number of classes)
# without loss of generality, assume x_train and x_validation have shape [n_example, n_feature].

# building an array of 'n_class' RBF networks for classification with 'n_class' classes
# 'X' is the set of centers of the networks, can also be a subset of x_train
# currently only Gaussian kernel (exp(-||x - y||^2/(2 * sigma ** 2))) is supported

net = feedforward() 
net.add_layer(kFullyConnected(X=x_train, n_out=n_class, kernel='gaussian', sigma=1))

# or, we can make 'X' adaptive by setting it to be a set of learnable parameters
# 'p' controls the number of centers
# net.add_layer(kFullyConnected(X=torch.randn(p, x_train.size(1)), n_out=n_class, kernel='gaussian', sigma=1, trainable_X=True))

# if your 'X' is a large sample, you might have trouble training this model
# on GPU. In that case, do
# layer = kFullyConnected(X=x_train, n_out=n_class, kernel='gaussian', sigma=1)
# net.add_layer(layer.to_ensemble(100))
# what the above two lines do is that they break 'X' into a few chunks of 100 examples (the last chunk may be smaller)
# and evaluate them in a sequential fashion without having to put them on 
# your GPU all together. Creating more chunks will reduce memory use but make the program slower since GPU likes 
# to compute things in parallel but not sequentially. This setting does not affect the output of the 
# network, only the way this output was computed.

# add optimizer, loss, and metric (for validation)
net.add_optimizer(
    torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    )
net.add_loss(torch.nn.CrossEntropyLoss())
net.add_metric(K.L0Loss()) # classification error (for validation)

# start training
net.fit(
    n_epoch=n_epoch,
    batch_size=batch_size,
    shuffle=shuffle, # shuffle data for each epoch
    X=x_train,
    Y=y_train,
    X_val=x_validation, # if you don't want to validate, just ignore 'X_val' and 'Y_val'
    Y_val=y_validation,
    val_window=val_window, # interval between two validations
    )

# test the trained model, print classification error
net.evaluate(X_test=x_test, Y_test=y_test, batch_size=batch_size, metric_fn=K.L0Loss())

```

## Build a kernel Multilayer Perceptron proposed in this paper (TDOO: url to be added)

## Build a neural-kernel hybrid network

## Train a kernelized network layer-wise for classification

## Wrap kerNET around any PyTorch object and use the helper functions to streamline your code
```python
import torch

from kernet.models.feedforward import feedforward

# build the classic LeNet-5
class LeNet5(torch.nn.Module):

    def __init__(self, in_channels, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=padding)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# wrap it with the 'feedforward' wrapper from kerNET
wrapper = feedforward()
net = LeNet5(in_channels=in_channels, padding=padding)

wrapper.add_layer(net)

# add optimizer, loss, and metric (for validation)
wrapper.add_optimizer(
    torch.optim.Adam(params=wrapper.parameters(), lr=lr, weight_decay=weight_decay)
    )
wrapper.add_loss(torch.nn.CrossEntropyLoss())
wrapper.add_metric(K.L0Loss()) # classification error (for validation)

# start training
wrapper.fit(
    n_epoch=n_epoch,
    batch_size=batch_size,
    shuffle=shuffle, # shuffle data for each epoch
    X=x_train,
    Y=y_train,
    X_val=x_validation, # if you don't want to validate, just ignore 'X_val' and 'Y_val'
    Y_val=y_validation,
    val_window=val_window, # interval between two validations
    )

# test the trained model, print classification error
wrapper.evaluate(X_test=x_test, Y_test=y_test, batch_size=batch_size, metric_fn=K.L0Loss())
```

