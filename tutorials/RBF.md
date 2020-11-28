A [radial basis function (RBF) network](https://en.wikipedia.org/wiki/Radial_basis_function_network), or in general, a kernel machine, can be implemented with ```kernet.layers.klinear.kLinear```. 
Specifying ```kernel='gaussian'``` makes the network an RBF network (special case of a kernel machine where the kernel involved is an RBF).
Full documentation can be found in [kernet/layers/klinear](../kernet/layers/klinear.py).

As an example, the following snippet initiates an RBF network with 10 output units using the Gaussian kernel with kernel width 1 and some user-specified centers.
```angular2
from kernet.layers.klinear import kLinear

my_centers = ...  # some data

net = kLinear(
    out_features=10,
    kernel='gaussian',
    evaluation='indirect',
    centers=my_centers,
    sigma=1
) 
```

## Memory Efficiency 

When using the kernel trick to approximate a kernel machine, [the representer theorem](https://en.wikipedia.org/wiki/Representer_theorem) necessitates basing the approximation on the full training set.
In practice, a naive implementation for such a scheme on a typical image dataset can require a daunting amount of memory.
In this case, use ```kernet.utils.networks.to_committee``` to convert your kernel machine into a memory-efficient version of itself (with the same parameters), and the amount of memory used can be controlled by specifying the expert size. 
More details on how we implement this feature are provided in the docstring of ```kernet.layers.klinear.kLinearCommittee```.