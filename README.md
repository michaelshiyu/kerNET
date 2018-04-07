# kerNET

kerNET is a simple, high-level, PyTorch-based API that helps you build kernel machine-powered connectionist models easily. It is based on [PyTorch](http://pytorch.org/) in order to make GPU acceleration possible (just like neural networks, these models require operations on large matrices).
kerNET is essentially PyTorch plus some extra layers that are building blocks for such models.
For convenience, a few higher-level model abstractions are also available, including MultiLayer Kernel Network (MLKN) proposed in [this paper](https://arxiv.org/abs/1802.03774).

Building a network with kernel machines here is as straightforward as building a neural network in PyTorch: you basically just need to read [this great tutorial](http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py).

For now, the project is very new and fairly unstable. We have much to add to it and we are also working on optimizing what's already there. Major functionalities have been tested (on CPU) under the following setup:
- Python 3
- PyTorch 0.3.1
- NumPy 1.14.1
- scikit-learn 0.19.1

Currently, the best way to work with this API is by forking it or via ```git clone```. Hope you enjoy it and any suggestion or contribution would be greatly appreciated!

# MultiLayer Kernel Network

A MLKN is an equivalent of a fully-connected, feedforward neural network in the universe of kernel machines. It works just like a [MultiLayer Perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron) except that under the hood, everything is now being powered by [kernel machines](https://en.wikipedia.org/wiki/Radial_basis_function_network).

In this repository, you will find a pre-built yet still highly customizable MLKN model. You can easily configure the size of the network and some other features just like using any other high-level neural network APIs. For training, besides all native methods of PyTorch, we have implemented [the proposed layerwise method](https://arxiv.org/abs/1802.03774) for you so that you only need to specify some hyperparameters in a few lines of codes to get things working. Some datasets used in the paper are also readily available for you to test the model out.

# Training a MLKN classifier layer-by-layer for [the Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)

Some imports and preprocessing on data to get things ready.
```
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
```
from models.mlkn import MLKNClassifier
mlkn = MLKNClassifier()
```

Let's implement a two-layer MLKN with 15 kernel machines on the first layer and ```n_class``` kernel machines on the second (because we will use cross-entropy as our loss function later and train the second layer as a RBFN). ```kerLinear``` is a ```torch.nn.Module``` object that represents a layer of kernel machines which use identical Gaussian kernels ```k(x, y) = e^(-1/2\sigma^2 ||x - y||^2_2)```. ```sigma``` controls the kernel width. One should always set the ```ker_dim``` parameter to the number of examples in your training set for the layer.
```
from layers.kerlinear import kerLinear
mlkn.add_layer(
    kerLinear(ker_dim=x_train.shape[0], out_dim=15, sigma=5, bias=True)
    )
mlkn.add_layer(
    kerLinear(ker_dim=x_train.shape[0], out_dim=n_class, sigma=.1, bias=True)
    )
```

Then add optimizer for each layer. This works with any ```torch.optim.Optimizer```. Each optimizer is in charge of one layer with the order of addition being the same with the order of layers, i.e., the first-added optimizer would be assigned to the first layer (layer closest to the input). For each optimizer, one can specify ```params``` to anything and it will be overridden to the weights of the correct layer automatically before the network is trained when ```fit``` is called.
```
mlkn.add_optimizer(
    torch.optim.Adam(params=mlkn.parameters(), lr=1e-3, weight_decay=0.1)
    )
mlkn.add_optimizer(
    torch.optim.Adam(params=mlkn.parameters(), lr=1e-3, weight_decay=.1)
    )
```

Specify loss function for the output layer, this works with any PyTorch loss function but let's use ```torch.nn.CrossEntropyLoss``` for this classification task.
```
mlkn.add_loss(torch.nn.CrossEntropyLoss())
```

Fit the model. For ```n_epoch```, one should pass a tuple of ```int```s with the first number specifying the number of epochs to train the first layer, etc. ```shuffle``` governs if the entire dataset is randomly shuffled at each epoch. If ```accumulate_grad``` is ```True```, the weights are only updated at each epoch instead of each minibatch using the accumulated gradient from all minibatches. If it is set to ```False```, there will be an update per minibatch.
```
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
```
y_pred = mlkn.predict(X_test=x_test, X=x_train, batch_size=15)
err = mlkn.get_error(y_pred, y_test)
print('error rate: {:.2f}%'.format(err.data[0] * 100))
```

This example is available at examples/mlkn_classifier.py. Some more classification datasets are there for you the try the model out.

# Lower-Level Kernel Machine-Based Objects

Apart from the pre-built, high-level models, we offer some basic building blocks to give you more freedom to customize.

(Tutorials will be added soon.)
