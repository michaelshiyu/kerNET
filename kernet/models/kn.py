# -*- coding: utf-8 -*-
# updated documentation Aug9_2018
import types

import torch
from torch.autograd import Variable

import kernet.backend as K
from kernet.layers.kerlinear import kerLinear
from kernet.layers.multicomponent import kerLinearEnsemble

torch.manual_seed(1234)

# TODO train the last layer in KNGreedy as an SVM (wrap sklearn.SVC?)
# TODO more cost functions for hidden layers in KNGreedy (Frobenius)
# TODO more kernels
# TODO ideal kernel matrix for more kernels
# TODO more flexibility in write_to (write settings of training, model details, etc.)

class baseKN(torch.nn.Module):
    """
    Base KN. Do not use.
    """
    def __init__(self):
        super().__init__()
        self._layer_counter = 0

    def add_layer(self, layer):
        """
        Add a layer to the model. A counter keeps record of the layers added.
        Layer indexing: the first layer added (layer 0) is the closest to 
        input. 

        Parameters
        ----------
        layer : a layer instance .
        """
        assert isinstance(layer, torch.nn.Module)
        self.add_module('layer'+str(self._layer_counter), layer)
        self._layer_counter += 1

    def add_loss(self, loss_fn):
        """
        Specify loss function for the last layer. This is the objective function
        to be minimized by the optimizer.

        This method assumes loss_fn is a callable torch loss function. It is 
        hard to perform a hard type check since there does not seem to be a 
        base class for all PyTorch loss classes.

        Parameters
        ----------
        loss_fn : callable torch loss function
        """
        # NOTE: CrossEntropyLoss (CrossEntropyLoss(output, y)), 
        # in PyTorch (as of 0.3.1), is
        # equivalent to NLLLoss(logsoftmax(output), y), where logsoftmax =
        # torch.nn.LogSoftmax(dim=1) for output Tensor of shape
        # (n_example, n_class). Base of the log functions in these operations
        # is e.
        self.add_module('output_loss_fn', loss_fn)

    def add_metric(self, metric_fn):
        """
        Specify a metric against which the model is evaluated. Primarily used 
        during training for validation.

        This method assumes metric_fn is a callable torch loss function. It is 
        hard to perform a hard type check since there does not seem to be a 
        base class for all PyTorch loss classes.

        Parameters
        ----------
        metric_fn : callable, torch loss function
        """
        setattr(self, '_metric_fn', metric_fn)

    def forward(self, x, upto=None, update_X=False):
        """
        Feedforward upto layer 'upto'. If 'upto' is not passed,
        this works as the standard forward function in PyTorch.

        Parameters
        ----------

        x : Tensor, shape (batch_size, dim)
            Batch to be evaluated.

        upto (optional) : int
            Index for the layer upto (and including) which we will evaluate
            the model. 0-indexed.

        update_X (optional) : bool
            Whether the X of each layer will be updated before passing x to the 
            model. Set to true if the parameters of the network have been 
            updated since last call with update_X set to true. 

            Will cause an error if set to true for layers that are not 
            kerLinear or kerLinearEnsemble.

        Returns
        -------
        y : Tensor, shape (batch_size, out_dim)
            Batch output.
        """
        if upto is not None: 
        # cannot use 'if upto' here since it is 0-indexed
        # and layer0 is the first layer
            assert 0<=upto<=self._layer_counter
            upto += 1
        else: upto = self._layer_counter # evaluate all layers instead

        y_previous = x
        for i in range(upto):
            layer = getattr(self, 'layer'+str(i))

            if update_X and i==0:
                if layer.__class__.__name__=='kerLinearStack': 
                    # for stack layer
                    # stack layer can only be the first layer in a KN
                    for j in range(layer._comp_counter):
                        # update X for each component in the stack
                        if j > 0:
                            comp = getattr(layer, 'comp'+str(j))
                            if comp.__class__.__name__=='kerLinearEnsemble': 
                            # for ensemble layer
                                for n in range(comp._comp_counter):
                                    # update X for each component in the ensemble
                                    comp_comp = getattr(comp, 'comp'+str(n))
                                    comp_comp.X = layer.forward(
                                        comp_comp.X_init, upto=j-1).detach()

                            elif comp.__class__.__name__=='kerLinear':
                                comp.X = layer.forward(
                                    comp.X_init, upto=j-1).detach()

            if update_X and i>0:
                # layer0.X does not need update unless the layer is a stack
                if layer.__class__.__name__=='kerLinearEnsemble': 
                    # for ensemble layer
                    for j in range(layer._comp_counter):
                        # update X for each component in the ensemble
                        comp = getattr(layer, 'comp'+str(j))
                        comp.X = self.forward(comp.X_init, upto=i-1).detach()

                elif layer.__class__.__name__=='kerLinear':
                    layer.X = self.forward(layer.X_init, upto=i-1).detach()
                    # if do not update using X_init, shape[1] of this
                    # data will not match data in previous layers, will cause
                    # issue in kerMap
            y = layer(y_previous)
            y_previous = y

        return y

    def evaluate(self, X_test, Y_test=None, layer=None, batch_size=None, 
        write_to=None, end=None):
        """
        Feed in test data and get output at a specified layer. This basically
        does the same thing as 'forward'. But 'evaluate' should be favored over 
        'forward' for test or validation since evaluate uses minimal memory and 
        is faster because it does not record anything for gradient calculations
        and it does not update X.

        Also if the entire network is evaluated, this function incorporates 
        the calculation of the given metric and prints the value. 
        If the metric used is L0Loss (classification error rate) 
        you can further store that value somewhere using the write_to option.

        Parameters
        ----------
        X_test : Tensor, shape (n_example, dim)
            Test data on which we want to evaluate the network.

        Y_test (optional) : Tensor, shape (n_example, y_dim)
            Labels. For calculating value of the metric.

        layer (optional) : int
            Output of this layer is returned. Layers are
            zero-indexed with the 0th layer being the one closest to the input.
            If this parameter is not passed, evaluate the output of the entire
            network instead.

        batch_size (optional) : int
            Split X_test into batches. If the size of X_test cannot be divided
            by batch_size, the last batch will be the last r examples in X_test,
            where r is the remainder of the division.

        write_to (optional) : str
            Address of the file to write the history of the L0Loss to. Each 
            write only append value of the metric to the file. It does not do
            anything else to the file for you. 

        end (optional) : str
            Trailing symbol for the write function at each write. 
            Default is "\n".

        Returns
        -------
        X_eval : Tensor, shape (n_example, layer_dim)
            Hidden representation of X_test at the given layer.
        """
        with torch.no_grad():
            if not batch_size or batch_size > X_test.shape[0]: 
                batch_size = X_test.shape[0]

            if layer is None: layer=self._layer_counter-1
            else: assert 0<=layer<=self._layer_counter-1

            layer, layer_index = getattr(self, 'layer'+str(layer)), layer
            # layer points to the actual layer instance, whereas layer_index
            # refers to its index, which is an integer

            i = 0 # batch index
            for x_test in K.get_batch(X_test, batch_size=batch_size):
                x_eval = self.forward(
                    x_test[0],
                    upto=layer_index,
                    update_X=False
                    )
                if i==0:
                    X_eval = x_eval
                else:
                    X_eval = torch.cat((X_eval, x_eval)) 
                    # concatenate output from each batch
                i += 1

            if layer_index == self._layer_counter-1 and Y_test is not None:
                # compute and print the metric if the entire model has been 
                # evaluated, only evaluate when Y_test is provided
                if self._metric_fn.__class__.__name__=='L0Loss':
                    # if metric is classification error rate, need to make 
                    # prediction first
                    _, Y_pred = torch.max(X_eval, dim=1)

                    print('{}: {:.3f}'.format(
                        'L0Loss (%)',
                        self._metric_fn(y_pred=Y_pred, y=Y_test)*100
                    ))

                    if write_to:
                        print('{}: {:.3f}'.format(
                            'L0Loss (%)',
                            self._metric_fn(y_pred=Y_pred, y=Y_test)*100
                        ), file=open(write_to,'a'), end=end)
                
                else:
                    print('{}: {:.3f}'.format(
                        self._metric_fn.__class__.__name__,
                        self._metric_fn(X_eval, Y_test).item()
                    ))

            return X_eval

    def fit(self):
        raise NotImplementedError('must be implemented by subclass')

    def save(self):
        pass
        # TODO: wrap native PyTorch support for save

class KN(baseKN):
    """
    A general KN model that does everything. Trained using backpropagation.
    """
    def add_optimizer(self, optimizer):
        """
        One optimizer for the entire model. But this of course supports those
        fancy per-parameter options from PyTorch.
        """
        assert isinstance(optimizer, torch.optim.Optimizer)
        setattr(self, 'optimizer', optimizer)

    def fit(
        self,
        n_epoch,
        X, Y,
        batch_size=None,
        shuffle=False,
        accumulate_grad=True,
        X_val=None,
        Y_val=None,
        val_window=30,
        verbose=True
        ):
        """
        Train the model on some data.

        Parameters
        ----------
        n_epoch : int
            The number of epochs to train the model.

        X : Tensor, shape (n_example, dim)
            Training set.

        Y : Tensor, shape (n_example, 1) or (n_example,)
            Target output of the training set.

        X_val (optional) : Tensor, shape (n_example, dim)
            Optional validation set.

        Y_val (optional) : Tensor, shape (n_example, 1) or (n_example,)
            Target output of the validation set.

        val_window (optional) : int
            The number of epochs between validation.

        batch_size (optional) : int
            Batch size for training. If the sample size is not divisible by the 
            batch size, the last batch would be the last r examples in the 
            training set, where r is the remainder of the division. 
            If not specified, use the entire sample.

        shuffle (optional) : bool
            Shuffle the data at each epoch.

        accumulate_grad (optional) : bool
            If set to true, accumulate gradient from each batch and only 
            update the weights after each epoch.

        verbose (optional) : bool
            If true, print value of the loss function at each epoch. If false
            and validation sets are provided, only print value of the validation 
            metric at each validation.
        """
        assert X.shape[0]==Y.shape[0]

        if not batch_size or batch_size > X.shape[0]: 
            batch_size = X.shape[0]
        n_batch = X.shape[0] // batch_size
        last_batch = bool(X.shape[0] % batch_size)

        # unfreeze the model
        for param in self.parameters(): param.requires_grad_(True) 
        for _ in range(n_epoch):
            __ = 0 # batch counter
            self.optimizer.zero_grad()

            for x, y in K.get_batch(
                X, Y,
                batch_size=batch_size,
                shuffle=shuffle
                ):

                # update X by default at the beginning of training
                update_X = True if _==0 and __==0 else False

                __ += 1

                output = self(x, update_X=update_X)

                loss = self.output_loss_fn(output, y)
                # L2 regulatization
                # is taken care of by setting the weight_decay param in the
                # optimizer

                if verbose:
                    print('epoch: {}/{}, batch: {}/{}, loss({}): {:.3f}'.format(
                        _+1, n_epoch, __, n_batch+int(last_batch),
                        self.output_loss_fn.__class__.__name__,
                        loss.item()
                        ))

                if not accumulate_grad:
                    # update weights at each batch
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # update X after a weight update
                    self.forward(x, update_X=True)

                else:
                    # accumulate gradient, do not update weights
                    loss.backward(retain_graph=True)

            if accumulate_grad:
                # update weights using the accumulated gradient from each batch
                self.optimizer.step()
                self.optimizer.zero_grad()
                # update X after a weight update
                self.forward(x, update_X=True)

            if X_val is not None and (_+1) % val_window==0:
                self.evaluate(X_test=X_val, Y_test=Y_val)

        if verbose: print('\n' + '#'*10 + '\n')

class KNGreedy(baseKN):
    """
    Base class for a greedy KN classifier. Do not use.
    
    1) kerLinear, kerLinearEnsemble and kerLinearStack are allowed as layers
    2) each of the above three objects is recognized as *one* layer
    3) kerLinearStack must be the first layer (closest to input)
    4) after a kerLinearStack, only kerLinear or kerLinearEnsemble can follow
    5) for NN-KN hybrid, build a separate torch.nn.Module and add as the first 
    layer
    """

    def __init__(self):
        super().__init__()
        self._optimizer_counter = 0

    def add_optimizer(self, optimizer):
        """
        One needs to designate an optimizer for
        each layer. The ordering for optimizers follows that of the layers. The
        number of optimizer added must match that of the layer otherwise an 
        error will be raised before training.

        User can pass any iterable to the non-optional 'params' parameter
        of the optimizer instance and this will be later overwritten to map 
        each optimizer to the correct layer.
        """
        assert isinstance(optimizer, torch.optim.Optimizer)
        setattr(self, 'optimizer'+str(self._optimizer_counter), optimizer)
        self._optimizer_counter += 1
        # optimizer indexing : optimizer 0 is the optimizer for layer 0

    def _compile(self):
        """
        Compile the model for training.
        """
        assert self._optimizer_counter==self._layer_counter
        # assign each optimizer to its layer
        for i in range(self._optimizer_counter):
            layer = getattr(self, 'layer'+str(i))
            optimizer = getattr(self, 'optimizer'+str(i))
            optimizer.param_groups[0]['params'] = list(layer.parameters())

        for param in self.parameters(): param.requires_grad_(False) 
        # freeze all layers. only unfreeze a layer when optimizing it. reduces
        # a lot of unnecessary gradient calculations

    def _fit_hidden(
        self,
        n_epoch,
        X, Y,
        n_group,
        batch_size=None,
        shuffle=False,
        accumulate_grad=True,
        cost='alignment',
        cluster_class=True,
        keep_grad=False,
        verbose=True
        ):
        """
        Train the hidden layers (all layers but the last).
        
        Parameters
        ----------
        n_epoch : iterable of ints
            The number of epochs for each hidden layer. The first integer matches 
            the layer closest to input and so on.
            At least one integer should be provided for each hidden layer. 
            Otherwise an error will be raised. If the length of the iterable is
            greater than the number of the layers, use n_epoch[:n_hiden_layer].

            Even if there is only one layer, this parameter must be an iterable 
            (may be of a scalar, e.g., (1,)).
            

        X : Tensor, shape (n_example, dim)
            Training set.

        Y : Tensor, shape (n_example, 1) or (n_example,)
            Categorical labels for the set.

        n_group : int
            The number of classes. Needed for the calculation of the target
            kernel matrix.

        batch_size (optional) : int
            Batch size for training. If the sample size is not divisible by the 
            batch size, the last batch would be the last r examples in the 
            training set, where r is the remainder of the division. 
            If not specified, use the entire sample.

        shuffle (optional) : bool
            Shuffle the data at each epoch.

        accumulate_grad (optional) : bool
            If set to true, accumulate gradient from each batch and only 
            update the weights after each epoch.

        verbose (optional) : bool
            If true, print value of the loss function at each epoch. 

        cost (optional) : str
            Cost function to use for the hidden layers. 'alignment' or 'MSE'.

        cluster_class (optional) : str
            Whether to enforce examples from the same class to be mapped closer
            in the RKHS. Default to True. The optimal representation does not 
            naturally require this, but by enforcing so, the upper layers can 
            be easily accelerated without much performance loss.

            This is ignored if cost is alignment.

        keep_grad (optional) : bool
            If True, will not zero grad after each grad update (which will lead
            to wrong results!). Set to true only when you want to check or 
            use the gradient for some reason.
        """

        assert cost in ['alignment', 'MSE']
        assert cluster_class in [True, False]
        
        # this model only supports hard class labels
        assert len(Y.shape) <= 2
        assert X.shape[0]==Y.shape[0]

        if not batch_size or batch_size > X.shape[0]: 
            batch_size = X.shape[0]
        n_batch = X.shape[0] // batch_size
        last_batch = bool(X.shape[0] % batch_size)

        if len(Y.shape)==1: Y=Y.view(-1, 1)
        # K.ideal_gram() requires label tensor to be of shape (n, 1)

        # default cost function for hidden layers is empirical alignment
        # can also use MSE (between two matrices)
        if cost=='alignment':
            loss_fn = torch.nn.CosineSimilarity() # equivalent to alignment
        elif cost=='MSE':
            loss_fn = torch.nn.MSELoss(size_average=True, reduce=True)

        for i in range(self._layer_counter-1):
            optimizer = getattr(self, 'optimizer'+str(i))
            next_layer = getattr(self, 'layer'+str(i+1))
            layer = getattr(self, 'layer'+str(i))

            # unfreeze this layer
            for param in layer.parameters(): param.requires_grad_(True)

            for _ in range(n_epoch[i]): 
                __ = 0 # batch counter
                optimizer.zero_grad()
                for x, y in K.get_batch(
                    X, Y,
                    batch_size=batch_size,
                    shuffle=shuffle
                    ):
                    # update X by default at the beginning of training
                    update_X = True if _==0 and __==0 else False
                    __ += 1

                    # calculate the target kernel matrix
                    if cluster_class is True:
                        ideal_gram = K.ideal_gram(y, y, n_group)
                    else:
                        ideal_gram, zero_mask = K.ideal_gram(
                            y, y, n_group, ignore_same_class=True
                            )
                    # convert to float type, required by CosineSimilarity
                    ideal_gram=ideal_gram.to(torch.float)

                    output = self(x, upto=i, update_X=update_X)

                    # calculate the actual kernel matrix of this layer
                    gram = K.kerMap(
                        output,
                        output,
                        next_layer.sigma
                        )
                    
                    # negative alignment
                    # L2 regulatization
                    # is taken care of by setting the weight_decay param in the
                    # optimizer
                    if cost=='alignment':
                        loss = -loss_fn(gram.view(1, -1), ideal_gram.view(1, -1))
                    elif cost=='MSE':
                        if cluster_class is False:
                            loss = \
                            loss_fn(zero_mask*gram, zero_mask*ideal_gram)
                            # y.shape[0]/zero_mask.sum() * \
                            # TODO should rescale the loss by dividing 
                            # n_pairs_of_examples_from_distinct_classes to make
                            # it \emph{mean} squared error
                            # mul by y.shape[0] is to undo the size_average of
                            # torch.nn.MSELoss
                        else:
                            loss = loss_fn(gram, ideal_gram)

                    if verbose:
                        if cost=='alignment':
                            print('epoch: {}/{}, batch: {}/{}, loss({}): {:.3f}'.format(
                                _+1, n_epoch[i], __, n_batch+int(last_batch),
                                'Alignment',
                                -loss.item()
                                ))
                        elif cost=='MSE':
                            print('epoch: {}/{}, batch: {}/{}, loss({}): {:.3f}'.format(
                                _+1, n_epoch[i], __, n_batch+int(last_batch),
                                'MSE',
                                loss.item()
                                ))

                    loss.backward()
                    # no need to update X at each weight update in layerwise 
                    # training since all layers prior to the layer being trained
                    # were already trained and fixed
                    if not accumulate_grad:
                        optimizer.step()
                        if not keep_grad:
                            optimizer.zero_grad()

                if accumulate_grad:
                    optimizer.step()
                    if not keep_grad:
                        optimizer.zero_grad()

            if verbose: print('\n' + '#'*10 + '\n')
            # freeze this layer again
            for param in layer.parameters(): param.requires_grad_(False) 

    def _fit_output(
        self,
        n_epoch,
        X, Y,
        batch_size=None,
        shuffle=False,
        accumulate_grad=True,
        X_val=None,
        Y_val=None,
        val_window=30,
        write_to=None,
        end=None,
        keep_grad=False,
        verbose=True
        ):
        """
        Train the last layer.

        Parameters
        ----------
        n_epoch : int
            The number of epochs to train this layer.

        X : Tensor, shape (n_example, dim)
            Training set.

        Y : Tensor, shape (n_example, 1) or (n_example,)
            Categorical labels for the training set.

        X_val (optional) : Tensor, shape (n_example, dim)
            Optional validation set.

        Y_val (optional) : Tensor, shape (n_example, 1) or (n_example,)
           Categorical labels for the validation set.

        val_window (optional) : int
            The number of epochs between validation.

        batch_size (optional) : int
            Batch size for training. If the sample size is not divisible by the 
            batch size, the last batch would be the last r examples in the 
            training set, where r is the remainder of the division. 
            If not specified, use the entire sample.

        shuffle (optional) : bool
            Shuffle the data at each epoch.

        accumulate_grad (optional) : bool
            If set to true, accumulate gradient from each batch and only 
            update the weights after each epoch.

        verbose (optional) : bool
            If true, print value of the loss function at each epoch. If false
            and validation sets are provided, only print value of the validation 
            metric at each validation.

        write_to (optional) : str
            Address of the file to write the history of the metric value to. Each 
            write only append value of the metric to the file. It does not do
            anything else to the file for you. Only write when validation
            data is provided and the metric is L0Loss.

        end (optional) : str
            Trailing symbol for the write function at each write. 
            Default is "\n".

        keep_grad (optional) : bool
            If True, will not zero grad after each grad update (which will lead
            to wrong results!). Set to true only when you want to check or 
            use the gradient for some reason.
        """

        # this model only supports hard class labels
        assert len(Y.shape) <= 2 
        assert X.shape[0]==Y.shape[0]

        if not batch_size or batch_size > X.shape[0]: 
            batch_size = X.shape[0]
        n_batch = X.shape[0] // batch_size
        last_batch = bool(X.shape[0] % batch_size)

        i = self._layer_counter-1
        optimizer = getattr(self, 'optimizer'+str(i))
        layer = getattr(self, 'layer'+str(i))

        # unfreeze this layer
        for param in layer.parameters(): param.requires_grad_(True) 
        for _ in range(n_epoch[i]):
            __ = 0 # batch counter
            optimizer.zero_grad()
            for x, y in K.get_batch(
                X, Y,
                batch_size=batch_size,
                shuffle=shuffle
                ):
                # update X by default at the beginning of training
                update_X = True if _==0 and __==0 else False
                __ += 1

                output = self.forward(x, upto=i, update_X=update_X)

                loss = self.output_loss_fn(output, y)
                # L2 regulatization
                # is taken care of by setting the weight_decay param in the
                # optimizer

                if verbose:
                    print('epoch: {}/{}, batch: {}/{}, loss({}): {:.3f}'.format(
                        _+1, n_epoch[i], __, n_batch+int(last_batch),
                        self.output_loss_fn.__class__.__name__,
                        loss.item()
                        ))

                loss.backward()
                # no need to update X at each weight update in layerwise 
                # training since all layers prior to the layer being trained
                # were already trained and fixed
                if not accumulate_grad:
                    optimizer.step()
                    if not keep_grad:
                        optimizer.zero_grad()

            if accumulate_grad:
                optimizer.step()
                if not keep_grad:
                    optimizer.zero_grad()

            if X_val is not None and (_+1) % val_window==0:
                self.evaluate(X_test=X_val, Y_test=Y_val, 
                    write_to=write_to, end=end
                    )

        if verbose: print('\n' + '#'*10 + '\n')
        # freeze this layer again
        for param in layer.parameters(): param.requires_grad_(False) 

class KNClassifier(KNGreedy):
    """
    KNClassifier implements a greedily-trained KN. This training algorithm
    can be tedious to implement in standard PyTorch since a lot of details 
    need to be taken care of so we build it for your convenience.

    If one wants a KN classifier trained with standard backpropagation,
    use KN class instead.
    """

    def fit(
        self,
        n_epoch,
        X, Y,
        n_class,
        batch_size=None,
        shuffle=False,
        accumulate_grad=True,
        X_val=None,
        Y_val=None,
        val_window=30,
        hidden_cost='alignment',
        cluster_class=True,
        write_to=None,
        end=None,
        keep_grad=False,
        verbose=True
        ):
        """
        Train the model on some data.

        Parameters
        ----------
        n_epoch : iterable of ints
            The number of epochs for each layer. The first integer matches 
            the layer closest to input and so on.
            At least one integer should be provided for each layer. 
            Otherwise an error will be raised. If the length of the iterable is
            greater than the number of the layers, use n_epoch[:n_layer].

            Even if there is only one layer, this parameter must be an iterable 
            (may be of a scalar, e.g., (1,)).

        X : Tensor, shape (n_example, dim)
            Training set.

        Y : Variable of shape (n_example, 1) or (n_example,)
            Categorical labels for the training set.

        n_class : int
            The number of classes.

        X_val (optional) : Tensor, shape (n_example, dim)
            Optional validation set.

        Y_val (optional) : Tensor, shape (n_example, 1) or (n_example,)
            Categorical labels for the validation set.

        val_window (optional) : int
            The number of epochs between validations.

        hidden_cost (optional) : str
            Cost function to use for the hidden layers. Default to 'alignment'.
            Can also be 'MSE'.

        cluster_class (optional) : str
            Whether to enforce examples from the same class to be mapped closer
            in the RKHS. Default to True. The optimal representation does not 
            naturally require this, but by enforcing so, the upper layers can 
            be easily accelerated without much performance loss.

        write_to (optional) : str
            Address of the file to write the history of the metric value to. Each 
            write only append value of the metric to the file. It does not do
            anything else to the file for you. Only write when validation
            data is provided and the metric is L0Loss.

        end (optional) : str
            Trailing symbol for the write function at each write. 
            Default is "\n".

        batch_size (optional) : int
            Batch size for training. If the sample size is not divisible by the 
            batch size, the last batch would be the last r examples in the 
            training set, where r is the remainder of the division. 
            If not specified, use the entire sample.

        shuffle (optional) : bool
            Shuffle the data at each epoch.

        accumulate_grad (optional) : bool
            If True, accumulate gradient from each batch and only update the
            weights after each epoch.

        keep_grad (optional) : bool
            If True, will not zero grad after each grad update (which will lead
            to wrong results!). Set to true only when you want to check or 
            use the gradient for some reason.
        """
        assert len(n_epoch) >= self._layer_counter
        self._compile()
        self._fit_hidden(
            n_epoch,
            X, Y,
            n_class,
            batch_size=batch_size,
            shuffle=shuffle,
            accumulate_grad=accumulate_grad,
            cost=hidden_cost,
            cluster_class=cluster_class,
            keep_grad=keep_grad,
            verbose=verbose
            )
        if verbose: print('\nHidden layers trained.\n')

        self._fit_output(
            n_epoch,
            X, Y,
            batch_size=batch_size,
            shuffle=shuffle,
            accumulate_grad=accumulate_grad,
            X_val=X_val,
            Y_val=Y_val,
            val_window=val_window,
            write_to=write_to,
            end=end,
            keep_grad=keep_grad,
            verbose=verbose
            )
        if verbose: print('\nOutput layer trained.\n')

if __name__=='__main__':
    pass
