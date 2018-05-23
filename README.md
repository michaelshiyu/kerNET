# kerNET

kerNET is a simple, high-level, PyTorch-based API that helps you build kernel machine-powered connectionist models easily. It is based on [PyTorch](http://pytorch.org/) in order to make GPU acceleration possible (just like neural networks, these models require operations on large matrices).
kerNET is essentially PyTorch plus some extra layers that are building blocks for such models.
For convenience, a few higher-level model abstractions are also available, including MultiLayer Kernel Network (MLKN) proposed in [this paper](https://arxiv.org/abs/1802.03774).

Building a network with kernel machines here is as straightforward as building a neural network in PyTorch: you basically just need to read [this great tutorial](http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py).

Dependencies:
- Python 3
- PyTorch 0.4.0

Currently, the best way to work with this API is by forking it or via ```git clone```. Hope you enjoy it and any suggestion or contribution would be greatly appreciated!

---------

## MultiLayer Kernel Network

A MLKN is an equivalent of a fully-connected, feedforward neural network in the universe of kernel machines. It works just like a [MultiLayer Perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron) except that under the hood, everything is now being powered by [kernel machines](https://en.wikipedia.org/wiki/Radial_basis_function_network).

In this repository, you will find a pre-built yet still highly customizable MLKN model. You can easily configure the size of the network and some other features just like using any other high-level neural network APIs. For training, besides all native methods of PyTorch, we have implemented [the proposed layerwise method](https://arxiv.org/abs/1802.03774) for you so that you only need to specify some hyperparameters in a few lines of codes to get things working. Some datasets used in the paper are also readily available for you to test the model out.

### training a MLKN classifier layer-by-layer for [the Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)

Some imports and preprocessings on data to get things ready.
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import torch
from torch.autograd import Variable

x, y = load_iris(return_X_y=True)

# normalize features to 0-mean and unit-variance
normalizer = StandardScaler()
x = normalizer.fit_transform(x)
n_class = int(np.amax(y) + 1)

# convert numpy data into torch objects
dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
X = Variable(torch.from_numpy(x).type(dtype), requires_grad=False)
Y = Variable(torch.from_numpy(y).type(dtype), requires_grad=False)

# randomly permute data
new_index = torch.randperm(X.shape[0])
X, Y = X[new_index], Y[new_index]

# split data evenly into training and test
index = len(X)//2
x_train, y_train = X[:index], Y[:index]
x_test, y_test = X[index:], Y[index:]
```

MLKNClassifier is a pre-built MLKN model with layerwise training already configured. It is implemented for classification with an arbitrary number of classes.
```python
from models.mlkn import MLKNClassifier
mlkn = MLKNClassifier()
```

Let's implement a two-layer MLKN with 15 kernel machines on the first layer and ```n_class``` kernel machines on the second (because we will use cross-entropy as our loss function later and train the second layer as a RBFN). ```kerLinear``` is a ```torch.nn.Module``` object that represents a layer of kernel machines which use identical Gaussian kernels ```k(x, y) = exp(-||x-y||_2^2 / (2 * sigma^2))```. ```sigma``` controls the kernel width. For ```X``` in the input layer, pass to it the random sample (usually the training set) you would like to center the kernel machines on, i.e., the set ```{x_i}``` in ```f(u) = ∑_i a_i k(x_i, u) + b```. This set is then an attribute of this layer object and can be visited as ```layer.X```. For non-input layers, pass to ```X``` the raw data you want to center the kernel machines on. At runtime, for layer ```n```, its ```X``` will be updated correctly to ```F_n-1(...(F_0(layer.X))...)```, where ```F_i``` is the mapping of the ```i```th layer.
```python
from layers.kerlinear import kerLinear
mlkn.add_layer(kerLinear(X=x_train, out_dim=15, sigma=5, bias=True))
mlkn.add_layer(kerLinear(X=x_train, out_dim=n_class, sigma=.1, bias=True))
```

For large datasets, it can be impossible operate on the entire training set due to insufficient memory. In this case, one can trade parallelism for memory sufficiency by breaking the training set into a few smaller subsets and center a separate ```kerLinear``` object on each subset. This is the same as breaking the Gram matrix into a bunch of submatrices and it will not make any difference to the numerical result. We have implemented a ```kerLinearEnsemble``` class and a helper function ```to_ensemble``` to make this process simpler. The script below will result in the same network and same calculations as the earlier method of adding layers. ```layer0``` and ```layer1``` are broken into many smaller networks each having 30 centers with weights and biases unchanged.
```python
from layers.kerlinear import kerLinear
from layers.ensemble import kerLinearEnsemble
import backend as K

ensemble = True
batch_size = 30

layer0 = kerLinear(X=x_train, out_dim=15, sigma=5, bias=True)
layer1 = kerLinear(X=x_train, out_dim=n_class, sigma=.1, bias=True)

if not ensemble:
    mlkn.add_layer(layer0)
    mlkn.add_layer(layer1)

else:
    # create equivalent ensemble layers so that large datasets can be fitted into memory
    mlkn.add_layer(K.to_ensemble(layer0, batch_size))
    mlkn.add_layer(K.to_ensemble(layer1, batch_size))
```

Then we add optimizer for each layer. This works with any ```torch.optim.Optimizer```. Each optimizer is in charge of one layer with the order of addition being the same with the order of layers, i.e., the first-added optimizer would be assigned to the first layer (layer closest to the input). For each optimizer, one can specify ```params``` to anything and it will be overridden to the weights of the correct layer automatically before the network is trained when ```fit``` is called. Let's use [Adam](https://arxiv.org/pdf/1412.6980.pdf) as the optimizer for this example. Note that for PyTorch optimizers, ```weight_decay``` is the l2-norm regularization coefficient.
```python
mlkn.add_optimizer(torch.optim.Adam(params=mlkn.parameters(), lr=1e-3, weight_decay=0.1))
mlkn.add_optimizer(torch.optim.Adam(params=mlkn.parameters(), lr=1e-3, weight_decay=.1))
```

Specify loss function for the output layer, this works with any PyTorch loss function but let's use ```torch.nn.CrossEntropyLoss``` for this classification task.
```python
mlkn.add_loss(torch.nn.CrossEntropyLoss())
```

Fit the model. For ```n_epoch```, one should pass a tuple of ```int``` with the first number specifying the number of epochs to train the first layer, etc. ```shuffle``` governs if the entire dataset is randomly shuffled at each epoch. If ```accumulate_grad``` is ```True```, the weights are only updated at each epoch instead of each minibatch using the accumulated gradient from all minibatches in that epoch. If it is set to ```False```, there will be an update per minibatch. Note that parameter ```X``` in ```fit``` is the training set you would like to train your model on, which can potentially be different from the set your kernel machines are centered on (parameter ```X``` when initializing a ```kerLinear``` object).
```python
mlkn.fit(
    n_epoch=(30, 30),
    batch_size=30,
    shuffle=True,
    X=x_train,
    Y=y_train,
    n_class=n_class,
    accumulate_grad=False
    )
```

Make a prediction on the test set and print error.
```python
y_pred = mlkn.predict(X_test=x_test, batch_size=15)
err = mlkn.get_error(y_pred, y_test)
print('error rate: {:.2f}%'.format(err.data[0] * 100))
```

This example is available at [examples/mlkn_classifier.py](https://github.com/michaelshiyu/kerNET/tree/master/examples). Some more classification datasets are there for you to try the model out.

---------

### training MLKN with backpropagation

In kerNET, we have also implemented a generic MLKN with maximal freedom to customization. Namely, it does not have greedy training pre-configured so it is easier to train it with the standard backpropagation together with some gradient-based optimization. Further, it is not defined to be a classifier, instead, it is a general-purpose learning machine: whatever you can do with MLP, you can do it with MLKN.

The generic MLKN works almost the same as MLKNClassifier. First we instantiate a model.
```python
import torch
from models.mlkn import MLKN
mlkn = MLKN()
```

Adding layers is the same as we did for MLKNClassifier. Ensemble layers are also supported.
```python
from layers.kerlinear import kerLinear
mlkn.add_layer(kerLinear(X=x_train, out_dim=15, sigma=5, bias=True))
mlkn.add_layer(kerLinear(X=x_train, out_dim=n_class, sigma=.1, bias=True))
```

For regression, the dimension of the output layer should be adjusted.
```python
mlkn.add_layer(kerLinear(ker_dim=x_train.shape[0], out_dim=y_train.shape[1], sigma=.1, bias=True))
```

Add an optimizer. This works with any ```torch.optim.Optimizer```. Unlike in the layerwise training case, here one optimizer is in charge of the training of the entire network since we are using backpropagation. But of course, this does not mean that all layers have to be trained under exactly the same setting: you could still specify [per-parameter options](http://pytorch.org/docs/master/optim.html) for each layer.
```python
mlkn.add_optimizer(torch.optim.Adam(params=mlkn.parameters(), lr=1e-3, weight_decay=0.1))
```

Specify a loss function. For classification, ```torch.nn.CrossEntropyLoss``` may be an ideal option whereas for regression, ```torch.nn.MSELoss``` is a common choice.
```python
mlkn.add_loss(torch.nn.CrossEntropyLoss())
```
Or, for regression,
```python
mlkn.add_loss(torch.nn.MSELoss())
```

Train the model and evaluate the output given some test set.
```python
mlkn.fit(
    n_epoch=30,
    batch_size=30,
    shuffle=True,
    X=x_train,
    Y=y_train,
    accumulate_grad=True
    )

y_raw = mlkn.evaluate(X_test=x_test, batch_size=15)
```

For classification, one may be interested in the error rate for this test set whereas for regression, MSE.

For classification,
```python
_, y_pred = torch.max(y_raw, dim=1)
y_pred = y_pred.type_as(y_test)
err = (y_pred!=y_test).sum().type(torch.FloatTensor).div_(y_test.shape[0])
print('error rate: {:.2f}%'.format(err.data[0] * 100))
```

For regression,
```python
mse = torch.nn.MSELoss()
print('mse: {:.4f}'.format(mse(y_raw, y_test).data[0]))
```

This example is available at [examples/mlkn_generic.py](https://github.com/michaelshiyu/kerNET/tree/master/examples). Some classification and regression datasets are there for you to try the model out.
