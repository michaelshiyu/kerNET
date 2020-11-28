In kerNET, there are three interfaces with different flexibility that provide access to our modular learning method.
1. [Easiest, but Minimum Flexibility](#easiest-but-minimum-flexibility)
1. [Full Customization Via Command Line Interface](#full-customization-via-command-line-interface)
1. [Using Component(s) From the Modular Learning Method](#using-components-from-the-modular-learning-method)

To test a trained model, see [Test a Trained Model](#test-a-trained-model). 

## 	Easiest, but Minimum Flexibility

If you only want to use the default set-up to train some classifiers on the datasets that we support, directly run the helper training scripts that we provide in [scripts/](../scripts/). 
For example, to train a ResNet-18 on CIFAR-10 with our modular method (as two modules), do 
```
cd scripts
./cifar10_modular.sh
```
To train a ResNet-18 on CIFAR-10 with end-to-end backpropagation as a baseline, do
```
cd scripts
./cifar10_e2e.sh
```

You can customize the training settings to some degree by modifying the options in the scripts.
By using the exact settings in those scripts, you should get performance on par with what's in the following table, which is what we got by running those scripts on our machine.
Do not expect exactly the same results since the models will not have exactly the same random initializations on your machine. 

|    | MNIST | Fashion-MNIST | CIFAR-10 | SVHN |
|---|---|---|---|---|
| end-to-end | 99.28 | 95.23 | 94.34 | 96.40 |
| modular | 99.32 | 95.31 | 93.96 | 96.73 |

Note that these hyperparameters provided in the scripts are not necessarily the best, the scripts are merely there for you to get a quick start.

## Full Customization Via Command Line Interface

To gain access to all the hyperparameters and settings that we support tuning, you may use our command line interface (the scripts in [scripts/](../scripts/) are in fact wrappers on top of this interface).
Specifically, to train a ResNet-18 on CIFAR-10 with our modular method (as two modules), do 
```angular2
python kernet/examples/modular_train.py --dataset cifar10 --model kresnet18 --n_parts 2 --loss xe --lr1 .1 --lr2 .1 --activation reapen --optimizer sgd --n_epochs1 200 --n_epochs2 50 --hidden_objective srs_upper_tri_alignment --in_channels 3 --batch_size 128  --save_dir my_checkpoint --n_val 5000
```

To see all the things that you can tune, do 
```angular2
python kernet/examples/modular_train.py -h
```

End-to-end training baselines can be obtained with [kernet/examples/train.py](../kernet/examples/train.py).

## Using Component(s) From the Modular Learning Method

If you want to use certain component(s) from our modular learning method, you can import the desired component(s) from kerNET into your own code. 
- [Proxy objectives](https://arxiv.org/pdf/2005.05541.pdf): These are the objective functions we use to train the hidden modules. Some reference implementations are in ```kernet.layers.loss```. 
- [Kernels induced by neural network nonlinearities](https://arxiv.org/pdf/2005.05541.pdf): Neural network nonlinearities such as tanh or ReLU can be used to induce kernel functions, which are then used to construct the proxy objectives and also enable us to view neural networks as kernel machines. These kernel functions can be accessed via  ```kernet.layers.kcore.Phi```.
- [Models](https://arxiv.org/pdf/2005.05541.pdf): The models in [kernet/models](../kernet/models/) whose names starts with a ```k``` are basically the same as the ones that do not start with a ```k```. The ```k``` models were implemented to work better with our modular learning pipeline.

## Test a Trained Model

Suppose your model is saved in ```my_checkpoint/```, to test it, do
```angular2
python kernet/examples/test.py --load_opt --opt_file my_checkpoint/opt.pkl --checkpoint_dir my_checkpoint
```

You can modify some settings during testing, to see all the things that you can tune, do 
```angular
python kernet/examples/test.py -h
```