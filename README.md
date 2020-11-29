[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/michaelshiyu/kerNET/master/LICENSE.md)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# kerNET

kerNET implements a modular training method for deep image classifiers.
In addition, kerNET can be used for implementing radial basis function networks or as a helpful wrapper that simplifies the training of classifiers.

See [Getting Started](#getting-started) for how to use kerNET.
See [References](#references) for the papers that proposed this modular learning method.

## Table of Contents
1. [Installation](#installation)
1. [Testing](#testing)
1. [Getting Started](#getting-started)
1. [License](#license) 
1. [References](#references)

## Installation
```angular2
pip install -r requirements.txt
pip install .

# or if you want developer install
pip install -e .
```

## Testing
We recommend using ```pytest``` for testing.
To run the test suites with ```pytest```, do
```angular2
pip install pytest
pytest test/
```
Note that some of the tests are computationally intensive as they involve training/testing networks and should therefore be executed on GPUs.

## Getting Started

kerNET is primarily for
- [modular learning for deep classifiers.](tutorials/MODULAR.md)

In the case where the network is trained as two modules, our modular learning method amounts to (1) training the input module with a special objective function called a "proxy objective", and then (2) freezing the input module and training the output module with a usual classification loss such as cross-entropy. 
The optimality of this method has been proved in certain (pretty general) settings in our papers (see [References](#references)).

kerNET is flexible.
In addition to the main functionality, kerNET
- [provides a memory-efficient implementation of radial basis function network;](tutorials/RBF.md)
- [can be used as a lightweight wrapper for classifier training (modular or end-to-end) that gives you access to a flexible, powerful pipeline via a command line interface.](tutorials/WRAPPER.md)

---

We currently support the following datasets and models. 

- Datasets
  - [MNIST](http://yann.lecun.com/exdb/mnist/)
  - [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
  - [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
  - [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
  - [SVHN](http://ufldl.stanford.edu/housenumbers/)
- Models
  - Classic kernel method-based models
    - [Radial basis function networks](https://en.wikipedia.org/wiki/Radial_basis_function_network), [kernel machines](https://en.wikipedia.org/wiki/Kernel_method)
  - Neural networks
    - [Fully-connected networks](https://en.wikipedia.org/wiki/Multilayer_perceptron)
    - [LeNet-5](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=726791)
    - [ResNets](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
  - Kernel method-based connectionist models
    - [Kernel networks](https://michaelshiyu.github.io/files/duan2020kernel.pdf)

You can add dataset and model by modifying [kernet/datasets](kernet/datasets/) and  [kernet/models](kernet/models/), respectively.

## License
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.

Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.

The code is released for academic research use only.

## References

The modular learning method implemented here is from our following two papers. 
BibTeX entries available in links below. 

- [Modularizing Deep Learning via Pairwise Learning With Kernels (IEEE Transactions on Neural Networks and Learning Systems, to appear)](https://michaelshiyu.github.io/publication/duan2020modularizing)
- [On Kernel Method-Based Connectionist Models and Supervised Deep Learning Without Backpropagation (Neural Computation, 2020)](https://michaelshiyu.github.io/publication/duan2020kernel)
