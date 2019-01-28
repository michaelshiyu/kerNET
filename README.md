# kerNET

**(Jan. 24, 2019) This repo is under active development. I am trying to make it easier to use and more memory efficient. Feel free to open an issue if you find something that doesn't work as expected. Also, I should remind you that some documentations are behind the actual code. I'm still working on that.**

kerNET is a [Keras](https://keras.io/)-like wrapper for PyTorch that makes it easier to build kernel networks and a layer-wise learning algorithm proposed in this paper (TDOO: url to be added).

Dependencies:
- Python 3.6
- PyTorch 1.0
- NumPy 1.15

To install, clone this repo to your local machine, go to the directory where the files live and do  ```python setup.py install```. 

Hope you enjoy this repo and any suggestion or contribution would be greatly appreciated!

Some simple tutorials are given below:

---------

## Build an RBF network

```python

net = FFC()
```

## Build a kernel Multilayer Perceptron proposed in this paper (TDOO: url to be added)

## Build a neural-kernel hybrid network

## Train a kernelized network layer-wise for classification

## Wrap kerNET around any PyTorch object and use the helper functions to streamline your code

