# kerNET

**(Jan. 27, 2019) This repo is under active development. I am trying to make it easier to use and more memory efficient. Feel free to open an issue if you find something that doesn't work as expected. Also, I should remind you that some documentations are behind the actual code. I'm still working on that.**

kerNET is a [Keras](https://keras.io/)-like wrapper for PyTorch that makes it easier to build kernel networks and a layer-wise learning algorithm proposed in [this paper](https://arxiv.org/abs/1802.03774).

Dependencies:
- Python 3.6
- PyTorch 1.0
- NumPy 1.15

To install, clone this repo to your local machine, go to the directory where the files live and do  ```python setup.py install```. 

Hope you enjoy this repo and any suggestion or contribution would be greatly appreciated!

Some simple tutorials are given below (they assume that you have some basic knowledge about PyTorch):

---------

## Build an [RBF network](https://en.wikipedia.org/wiki/Radial_basis_function_network) for classification

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
net.add_optimizer(torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay))
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

## Build a kernel Multilayer Perceptron proposed in [this paper](https://arxiv.org/abs/1802.03774)

[This paper](https://arxiv.org/abs/1802.03774) proposed a framework to "kernelize" any neural network (NN), i.e., substitute one or multiple nodes f(x) = \sigma(w^T x), where \sigma is a nonlinear activation function such as hyperbolic tangent, with kernel machines f(x) = <w, \phi(x)>, where \phi is a nonlinear mapping into an [RKHS](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space). For kernel machines, the w may not be accessible and is usually approximated using the [representer theorem](https://en.wikipedia.org/wiki/Representer_theorem). Kernelizing a single neuron would give the classic RBF network with one output node. And for an NN, one may freely choose the degree of kernelization: from one node only, to the entire network.

Below we demonstrate how to build a fully-kernelized MLP and train it with backpropagation using kerNET.

```python
import torch

import kernet.backend as K
from kernet.models.feedforward import feedforward
from kernet.layers.kernelized_layer import kFullyConnected

# say we would like to build a three-layer, fully-kernelized MLP
# all you need to change in the above tutorial is to add two more layers! The first layer you add would be the input layer and so on
# everything from the above tutorial still applies, such as the more memory-efficient ensemble mode
  
net = feedforward() 

net.add_layer(kFullyConnected(X=x_train, n_out=hidden_dim1, kernel='gaussian', sigma=3))
net.add_layer(kFullyConnected(X=x_train, n_out=hidden_dim2, kernel='gaussian', sigma=2))
net.add_layer(kFullyConnected(X=x_train, n_out=n_class, kernel='gaussian', sigma=1))

# note that you can later access these layers by calling on them, for example, the input layer can be accessed as 
input_layer = net.layer0 # the layers are zero-indexed

# for non-input layers, their centers 'X' actually require some special manipulations. You can refer to the paper for more details. For example, for the layer right after the input layer, its 'X' should be input_layer(x_train) and should be updated every time the parameters of the input layer change. But you do not need to worry about this as kerNET would take care of it in the background
```

## Build a neural-kernel hybrid network

Now we demonstrate how to only kernelize a part of a given NN. We use LeNet-5 as an example and kernelize the output layer.

```python
class LeNet5_minus_output_layer(torch.nn.Module):

    def __init__(self, in_channels, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=padding)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10) # we don't need the output layer defined here

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = feedforward()

layer0 = LeNet5_minus_output_layer(in_channels=in_channels, padding=padding)
layer1 = kFullyConnected(X=x_train, n_out=n_class, kernel='gaussian', sigma=1)

net.add_layer(layer0)
net.add_layer(layer1)
```

## Train a kernelized network layer-wise for classification

The main advantage of kernelizing a feedforward NN is that there is a way to train the resulting network layer-by-layer without backpropagation for classification. First note that why people need backpropagation to train a deep network is because there is no target information available for the hidden layers. For the kernelized models, on the other hand, the optimal target for each hidden layer that minimizes the objective function of the network can be explicitly characterized to some degree, as detailed in the paper. 

To train the earlier kernelized LeNet-5 layer-wise, do
```python
net = greedyFeedforward() # instantiate this class if you want to train your network layer-wise

... # add the layers as before

# add one optimizer for each layer. You can pass anything to the 'params' argument and kerNET would later assign each optimizer to the layer it should be working on
net.add_optimizer(torch.optim.Adam(params=net.parameters(), lr=lr0, weight_decay=weight_decay0))
net.add_optimizer(torch.optim.Adam(params=net.parameters(), lr=lr1, weight_decay=weight_decay1))

# loss function to train layer0 and metric function for validation
# below gives the alignment loss in the paper. You can substitute CosineSimilarity() by MSELoss(size_average=True, reduce=True) or L1Loss(size_average=True, reduce=True) to use the L^2 or L^1 loss in the paper, respectively
net.add_loss(torch.nn.CosineSimilarity())
net.add_metric(torch.nn.CosineSimilarity())

# loss and metric for layer1
net.add_loss(torch.nn.CrossEntropyLoss())
net.add_metric(K.L0Loss())

# layer1.phi is just the \phi function associated with the kernel of layer1. This determines how the kernel matrix of layer0 is computed. You can check the paper for more details (this kernel matrix is called G_0 in the paper). In short, suppose you have an l-layer model to train layer-wise, you should add layer_{i+1}.phi as the critic of layer_i for i = 1, 2, ..., l-1
net.add_critic(layer1.phi)

# start training
net.fit(
    n_epoch=(epo0, epo1), # epoch to train for 
    batch_size=batch_size,
    shuffle=shuffle,
    X=x_train,
    Y=y_train,
    n_class=n_class, # this information is needed to compute G^\star (see the paper for more information on this matrix)
    accumulate_grad=accumulate_grad,
    X_val=x_val,
    Y_val=y_val,
    val_window=val_window,
    )

net.evaluate(X_test=x_test, Y_test=y_test, batch_size=batch_size, metric_fn=K.L0Loss())
```

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

# instantiate a LeNet-5 and wrap it with the 'feedforward' wrapper from kerNET
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

