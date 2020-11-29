The main entry point for training models in kerNET is [kernet/examples/train.py](../kernet/examples/train.py) for end-to-end training, and [kernet/examples/modular_train.py](../kernet/examples/modular_train.py) for modular training.
Full documentation can be accessed by, e.g., 
```angular2
python kernet/examples/train.py -h
```

kerNET packs many useful features into its training pipeline.
And you have full access via the command line interface without having to modify the codebase.
We discuss some particularly helpful knobs here.

### Specify Model and Dataset

All supported models and datasets can be selected using ```--model``` and ```--dataset```.

More details on the models can be found in [here](MODULAR.md) (specifically, under Using Component(s) From the Modular Learning Method).

### Specify Training Settings

You can full control over training.
The command line interface allows you to choose data augmentation, loss function, optimizer, learning rate, number of epochs, weight decay, batch size, and so on.

You can also train with a randomly-chosen subset and keep using the same subset across multiple training sessions.

In modular training, the number of modules that you want to train your model as can be controlled via ```--n_parts```.
But keep in mind that if you cut the model too fine, the modules will not have enough capacity to learn well. 

### Validation, Model Saving and Loading

The size of the validation set can be specified using ```--n_val```.
This set will be a randomly-selected subset of the training set.
To use the same validation set across multiple training sessions, specify ```--dataset_rand_idx```.

By default, the model that performs the best on the validation set will be saved in each training session.
You may choose to not use a validation set and force the trainer to save your model at each epoch using ```--always_save```. 

To load a saved model, use ```--load_model``` and specify ```--checkpoint_dir```.

To load options specified for some other training session, use ```--load_opt``` and specify ```--opt_file```.
These loaded options will override the default options, and will be overwritten by your specified options.

### Logging

[Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) logging is available by specifying ```--tf_log```.

All training options and logs will also be saved in ```opt.txt``` and ```train.log```.

You can control the level of detailedness of the logging information that will be printed to the console via ```--log_level```.

Unless you select debug mode or other lower-level logging modes, a progress bar will show you some helpful summary statistics during training.
An example of this progress bar is given in the following figure.
<div align="center">
  <img src="https://github.com/michaelshiyu/kerNET/blob/master/tutorials/progress_bar.png"/>
</div>
In debug mode and other lower-level logging modes, statistics from each batch will be printed in full, which lets you see exactly what's going on in each batch but makes it difficult to have a high-level view on training.

### Test a Trained Model

See the section under the same name in [tutorials/MODULAR.md](MODULAR.md). 