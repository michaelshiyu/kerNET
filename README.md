# MultiLayer Kernel Network
# PyTorch-based high-level API for easy implementations of MultiLayer Kernel Network (https://arxiv.org/pdf/1802.03774.pdf).

A MultiLayer Kernel Network (MLKN) is an equivalent of a fully-connected, feedforward neural network in the universe of kernel machines. It works just like a MultiLayer Perceptron (MLP) (https://en.wikipedia.org/wiki/Multilayer_perceptron) except that under the hood, everything is now being powered by kernel machines (we need a good and easy-to-understand explanation of what a kernel machine is). 

In this repository, you will find a high-level API that allows you to build MLKN models easily. It is based on PyTorch for quick access to GPU acceleration (just like neural networks, MLKN requires operations on large matrices). Building a MLKN here is as straightforward as building a neural network in PyTorch: you basically just need to read this great tutorial (http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py).

Although, because of some uniqle properties of MLKN, not all functionalities from PyTorch are compatible. Hence we give another tutorial here (which is very easy to understand once you have understood the one linked above). 

---------
Tutorial needed.
---------
