# kerNET

kerNET is a simple, high-level, PyTorch-based API that helps you build kernel machine-powered connectionist models easily. It is based on [PyTorch](http://pytorch.org/) in order to make GPU acceleration possible (just like neural networks, these models require operations on large matrices).
kerNET is essentially PyTorch plus some extra layers that are building blocks for such models.
For convenience, a few higher-level model abstractions are also available, including MultiLayer Kernel Network (MLKN) proposed in [this paper](https://arxiv.org/pdf/1802.03774.pdf).

Building a network with kernel machines here is as straightforward as building a neural network in PyTorch: you basically just need to read [this great tutorial](http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py).

# MultiLayer Kernel Network

A MLKN is an equivalent of a fully-connected, feedforward neural network in the universe of kernel machines. It works just like a [MultiLayer Perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron) except that under the hood, everything is now being powered by [kernel machines](https://en.wikipedia.org/wiki/Radial_basis_function_network).

In this repository, you will find a pre-built yet still highly customizable MLKN model. You can easily configure the size of the network and some other features just like using any other high-level neural network APIs. For training, besides all native methods of PyTorch, we have implemented [the proposed layerwise method](https://arxiv.org/pdf/1802.03774.pdf) for you so that you only need to specify some hyperparameters in a few lines of codes. Some datasets used in the paper are also readily available for you to test the model out.

For an example, see this tutorial.

(add tutorial)

# Lower-Level Kernel Machine-Based Objects

Apart from the pre-built, high-level models, we offer some basic building blocks to give you more freedom to customize.

(add tutorial)
