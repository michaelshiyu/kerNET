#!/usr/bin/env python

import types, gc, sys

import torch

import kernet.backend as K
from kernet.layers.kernelized_layer import _kernelizedLayer, kFullyConnected, kFullyConnectedEnsemble, kFullyConnectedStack

# TODO tests
# TODO documentation is behind code
# TODO more flexibility in write_to (write settings of training, model details, etc.)
# TODO add support for trainable_X

class _baseFeedforward(torch.nn.Module):
    """
    Base feedforward. Do not use.
    """
    def __init__(self):
        super().__init__()
        self._layer_counter = 0
        self._print_tensors = False

    def add_layer(self, layer):
        """
        Add a layer to the model. A counter keeps record of the layers added.
        Layer indexing: the first layer added (layer 0) is the closest to 
        input. 

        Parameters
        ----------
        layer : a torch.nn.Module layer instance.
        """
        assert isinstance(layer, torch.nn.Module)
        self.add_module('layer{0}'.format(self._layer_counter), layer)
        self._layer_counter += 1

    def add_loss(self, loss_fn):
        """
        Specify loss function for the last layer. This is the objective function
        to be minimized by the optimizer.

        Parameters
        ----------
        loss_fn : callable torch loss function
        """
        assert isinstance(loss_fn, (
                torch.nn.modules.loss._Loss,
                torch.nn.modules.distance.CosineSimilarity
                ))
        self.add_module('loss_fn', loss_fn)

    def add_metric(self, metric_fn):
        """
        Specify a metric against which the model is evaluated. Primarily used 
        during training for validation.

        Parameters
        ----------
        metric_fn : callable, torch loss function
        """
        assert isinstance(metric_fn, torch.nn.modules.loss._Loss)
        setattr(self, 'metric_fn', metric_fn)

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

            This parameter has no effect on NN layers.

        Returns
        -------
        y : Tensor, shape (batch_size, out_dim)
            Batch output.
        """
        if upto is not None: 
        # cannot use 'if upto' here since it is 0-indexed
        # and layer0 is the first layer
            assert 0<=upto<=self._layer_counter
            upto += 1 # adjust indexing
        else: upto = self._layer_counter # evaluate all layers instead

        y_previous = x
        original_update_X = update_X # store original setting so that upstream nn layers won't override update_X for kn layer
        for i in range(upto):
            layer = getattr(self, 'layer'+str(i))

            # only need to update X for kernel layers
            if not isinstance(layer, _kernelizedLayer): update_X=False
            if isinstance(layer, _kernelizedLayer): update_X=original_update_X

            if update_X and i==0:
                if isinstance(layer, kFullyConnectedStack): 
                    # for stack layer
                    # stack layer can only be the first layer in a KN
                    for j in range(layer._comp_counter):
                        # update X for each component in the stack
                        if j > 0:
                            comp = getattr(layer, 'comp'+str(j))
                            if isinstance(comp, kFullyConnectedEnsemble): 
                            # for ensemble layer
                                for n in range(comp._comp_counter):
                                    # update X for each component in the ensemble
                                    comp_comp = getattr(comp, 'comp'+str(n))
                                    comp_comp.X = layer(
                                        comp_comp.X_init, upto=j-1).detach()

                            elif isinstance(comp, kFullyConnected):
                                comp.X = layer(
                                    comp.X_init, upto=j-1).detach()

            if update_X and i>0:
                # layer0.X does not need update unless the layer is a stack
                if isinstance(layer, kFullyConnectedEnsemble):  
                    # for ensemble layer
                    for j in range(layer._comp_counter):
                        # update X for each component in the ensemble
                        comp = getattr(layer, 'comp'+str(j))
                        comp.X = self(comp.X_init, upto=i-1).detach()

                elif isinstance(layer, kFullyConnected):
                    layer.X = self(layer.X_init, upto=i-1).detach()
                    # if do not update using X_init, shape[1] of this
                    # data will not match data in previous layers, will cause
                    # issue in kerMap
            y = layer(y_previous)
            y_previous = y

        return y

    def get_repr(self, data_loader, layer=None):
        '''
        forward method with support for PyTorch's data_loader.
        '''
        pass

    def evaluate(self, test_loader, metric_fn=None, critic=None, 
        hidden_val=False, n_class=None, layer=None, batch_size=None, 
        write_to=None, end=None):
        """
        Similar to forward. But never updates X. For inference.

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

        metric_fn :
            Metric function for validation.

        hidden_val : bool
            Whether this call is for hidden layer validation. If yes, compute ideal kmtrx in this function.

        critic:
            Kernel that will be used to compute G_i and the ideal kernel matrix G^\star as defined in the paper.

        Returns
        -------
        X_eval : Tensor, shape (n_example, layer_dim)
            Hidden representation of X_test at the given layer.
        """
        with torch.no_grad():

            if layer is None: layer=self._layer_counter-1
            else: assert 0<=layer<=self._layer_counter-1

            layer, layer_index = getattr(self, 'layer'+str(layer)), layer
            # layer points to the actual layer instance, whereas layer_index
            # refers to its index, which is an integer

            assert metric_fn is not None

            # TODO require metric_fn do not average over minibatch

            n_test = 0
            total_loss = 0
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(next(self.parameters()).device),\
                y_test.to(next(self.parameters()).device) # FIXME model sharding

                x_eval = self(
                    x_test,
                    upto=layer_index,
                    update_X=False
                    )

                if hidden_val==False:
                    # if not trying to get loss value from a hidden layer
                    if isinstance(metric_fn, K.L0Loss):
                        # if metric is classification error rate, need to make 
                        # prediction first
                        _, y_pred = torch.max(x_eval, dim=1)
                        batch_loss = metric_fn(y_pred=y_pred, y=y_test)
                        
                    else:
                        batch_loss = metric_fn(x_eval, y_test)
                    
                    n_test += x_test.size(0)
                    total_loss += batch_loss.item()
                else: # TODO test
                    # TODO may take a lot of memory even under torch.no_grad=True?
                    assert not isinstance(metric_fn, K.L0Loss) 
                    assert critic is not None and n_class is not None
                    # hidden layer does not validate w.r.t. L0Loss

                    if len(y_test.shape)==1: y_test=y_test.view(-1, 1)

                    ideal_kmtrx = critic.get_ideal_kmtrx(y_test, y_test, n_class=n_class) # val.n_class must be the same as train.n_class

                    # convert to float type, required by CosineSimilarity
                    ideal_kmtrx=ideal_kmtrx.to(torch.float)

                    # calculate the actual kernel matrix of this layer
                    kmtrx = critic.get_kmtrx(x_eval, x_eval)
                    
                    batch_loss = metric_fn(kmtrx.view(1, -1), ideal_kmtrx.view(1, -1))

                    n_test += x_test.size(0)
                    total_loss += batch_loss.item()

            loss = total_loss / n_test
            print('{}: {:.3f}'.format(
                metric_fn.__class__.__name__,
                loss
            ))
            if write_to:
                print('{}: {:.3f}'.format(
                    metric_fn.__class__.__name__,
                    loss
                ), file=open(write_to,'a'), end=end)
            return loss
                    

    def fit(self):
        raise NotImplementedError('must be implemented by subclass')

class feedforward(_baseFeedforward):
    """
    # TODO change the name to feedforward

    A general feedforward model that does everything. 
    Wrap this around any feedforward network in PyTorch, then you can use its helper functions such as fit, evaluate, etc. to streamline your code.
    Trained with backpropagation.
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
        train_loader,
        # batch_size=None,
        # shuffle=False,
        accumulate_grad=True,
        # X_val=None,
        # Y_val=None,
        val_loader=None,
        val_window=30,
        verbose=True,
        write_to=None,
        end=None
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
        '''
        assert X.shape[0]==Y.shape[0]

        if not batch_size or batch_size > X.shape[0]: 
            batch_size = X.shape[0]
        n_batch = X.shape[0] // batch_size
        last_batch = bool(X.shape[0] % batch_size)
        '''

        # unfreeze the model
        for param in self.parameters(): param.requires_grad_(True) 
        for _ in range(n_epoch):
            self.optimizer.zero_grad()

            for __, (x, y) in enumerate(train_loader):
                x, y = x.to(next(self.parameters()).device),\
                y.to(next(self.parameters()).device) # FIXME model sharding

                # update X by default at the beginning of training
                update_X = True if _==0 and __==0 else False

                output = self(x, update_X=update_X)

                loss = self.loss_fn(output, y) / x.size(0)

                if verbose:
                    print('epoch: {}/{}, batch: {}, loss({}): {:.3f}'.format(
                        _+1, n_epoch, __,
                        self.loss_fn.__class__.__name__,
                        loss.item()
                        ))
                loss.backward()
                if not accumulate_grad:
                    # update weights at each batch
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # update X after a weight update
                    self(x, update_X=True)

                # else:
                    # loss.backward(retain_graph=True)

            if accumulate_grad:
                # update weights using the accumulated gradient from each batch
                self.optimizer.step()
                self.optimizer.zero_grad()
                # update X after a weight update
                self(x, update_X=True)


            # validate
            
            if val_loader is not None and (_+1) % val_window==0:
                
                assert self.metric_fn is not None
                self.evaluate(test_loader=val_loader, 
                    write_to=write_to, end=end, metric_fn=self.metric_fn
                    )

        if verbose: print('\n' + '#'*10 + '\n')

class _greedyFeedforward(_baseFeedforward):
    """
    Base class for a greedy KN classifier. Do not use.
    
    The upper kernelized, fully-connected layers can be trained layer-wise.
    The input layer can be anything, allowing for NN-KN hybrid architectures.
    """

    def __init__(self):
        super().__init__()
        self._optimizer_counter = 0
        self._loss_fn_counter = 0
        self._metric_fn_counter = 0
        self._critic_counter = 0

    def add_critic(self, critic):
        '''
        Indexing follows that of the layers: 0=input layer
        '''
        assert isinstance(critic, K.Phi)
        setattr(self, 'critic'+str(self._critic_counter), critic)
        self._critic_counter += 1

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

    def add_loss(self, loss_fn):
        """
        Specify loss functions. This is the objective function
        to be minimized by the optimizer for each layer.
        The ordering for loss functions follows that of the layers.

        Parameters
        ----------
        loss_fn : callable torch loss function
        """
        assert isinstance(loss_fn, (
                torch.nn.modules.loss._Loss,
                torch.nn.modules.distance.CosineSimilarity
                ))
        setattr(self, 'loss_fn'+str(self._loss_fn_counter), loss_fn)
        self._loss_fn_counter += 1

    def add_metric(self, metric_fn):
        """
        Specify metric functions. This is the objective function
        to be used for validation for each layer.
        The ordering for loss functions follows that of the layers.

        Parameters
        ----------
        metric_fn : callable torch loss function
        """

        assert isinstance(metric_fn, (
                torch.nn.modules.loss._Loss,
                torch.nn.modules.distance.CosineSimilarity
                ))
        setattr(self, 'metric_fn'+str(self._metric_fn_counter), metric_fn)
        self._metric_fn_counter += 1

    def _compile(self):
        """
        Compile the model for training.
        """
        # output layer doesn't need a critic
        assert self._optimizer_counter==self._layer_counter==\
            self._loss_fn_counter==self._metric_fn_counter==\
            self._critic_counter+1 

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
        n_class,
        batch_size=None,
        shuffle=False,
        accumulate_grad=True,
        X_val=None,
        Y_val=None,
        val_window=30,
        val_batch_size=300,
        write_to=None,
        end=None,
        keep_grad=False,
        verbose=True,
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

        n_class : int
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

        model (optional) : str
            kMLP or MLP.
        """
        
        # this model only supports hard class labels
        assert len(Y.shape) <= 2
        assert X.shape[0]==Y.shape[0]

        if not batch_size or batch_size > X.shape[0]: 
            batch_size = X.shape[0]
        n_batch = X.shape[0] // batch_size
        last_batch = bool(X.shape[0] % batch_size)

        if len(Y.shape)==1: Y=Y.view(-1, 1)
        # K.ideal_kmtrx() requires label tensor to be of shape (n, 1)

        for i in range(self._layer_counter-1):
            optimizer = getattr(self, 'optimizer'+str(i))
            loss_fn = getattr(self, 'loss_fn'+str(i))
            metric_fn = getattr(self, 'metric_fn'+str(i))
            critic = getattr(self, 'critic'+str(i))
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

                    ideal_kmtrx = critic.get_ideal_kmtrx(y, y, n_class=n_class)

                    # convert to float type, required by CosineSimilarity
                    ideal_kmtrx=ideal_kmtrx.to(torch.float)

                    output = self(x, upto=i, update_X=update_X)

                    # calculate the actual kernel matrix of this layer
                    kmtrx = critic.get_kmtrx(output, output)
                    
                    # negative alignment
                    # L2 regulatization
                    # is taken care of by setting the weight_decay param in the
                    # optimizer
                    if loss_fn.__class__.__name__=='CosineSimilarity':
                        loss = -loss_fn(kmtrx.view(1, -1), ideal_kmtrx.view(1, -1))
                    else:

                        loss = loss_fn(kmtrx.view(-1, 1), ideal_kmtrx.view(-1, 1))

                    if verbose:
                        if loss_fn.__class__.__name__=='CosineSimilarity':
                            print('epoch: {}/{}, batch: {}/{}, loss({}): {:.3f}'.format(
                                _+1, n_epoch[i], __, n_batch+int(last_batch),
                                'Alignment',
                                -loss.item()
                                ))
                        else:
                            print('epoch: {}/{}, batch: {}/{}, loss({}): {:.3f}'.format(
                                _+1, n_epoch[i], __, n_batch+int(last_batch),
                                loss_fn.__class__.__name__,
                                loss.item()
                                ))

                    # no need to update X at each weight update in layerwise 
                    # training since all layers prior to the layer being trained
                    # were already trained and fixed
                    
                    loss.backward()
                    
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
                        write_to=write_to, end=end, metric_fn=metric_fn, critic=critic, hidden_val=True,
                            layer=i, n_class=n_class, batch_size=val_batch_size
                    )


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
        val_batch_size=300,
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
        loss_fn = getattr(self, 'loss_fn'+str(i))
        metric_fn = getattr(self, 'metric_fn'+str(i))

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
                
                output = self(x, upto=i, update_X=update_X)

                if loss_fn.__class__.__name__=='MarginRankingLoss': # binary hinge loss
    
                    ones = torch.ones_like(y)
                    nega_ones = torch.ones_like(y) * (-1)
                    y = torch.where(y==1, ones, nega_ones)

                    loss = loss_fn(output.view(-1,), torch.zeros_like(output, dtype=torch.float).view(-1,), y.to(torch.float).view(-1,))
                else:
                    loss = loss_fn(output, y)
                # L2 regulatization
                # is taken care of by setting the weight_decay param in the
                # optimizer

                if verbose:
                    print('epoch: {}/{}, batch: {}/{}, loss({}): {:.3f}'.format(
                        _+1, n_epoch[i], __, n_batch+int(last_batch),
                        loss_fn.__class__.__name__,
                        loss.item()
                        ))

                # no need to update X at each weight update in layerwise 
                # training since all layers prior to the layer being trained
                # were already trained and fixed
                loss.backward()
                
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
                    write_to=write_to, end=end, metric_fn=metric_fn, batch_size=val_batch_size
                    )
                
            


        if verbose: print('\n' + '#'*10 + '\n')
        # freeze this layer again
        for param in layer.parameters(): param.requires_grad_(False) 

class greedyFeedforward(_greedyFeedforward):
    """
    KNClassifier implements a greedily-trained KN. This training algorithm
    can be tedious to implement in standard PyTorch since a lot of details 
    need to be taken care of so we build it for your convenience.

    If one wants a KN classifier trained with standard backpropagation,
    use Feedforward class instead.
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
        val_batch_size=300,
        write_to=None,
        end=None,
        keep_grad=False,
        verbose=True,
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
            X_val=X_val,
            Y_val=Y_val,
            val_window=val_window,
            val_batch_size=val_batch_size,
            write_to=write_to,
            end=end,
            keep_grad=keep_grad,
            verbose=verbose,
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
            val_batch_size=val_batch_size,
            write_to=write_to,
            end=end,
            keep_grad=keep_grad,
            verbose=verbose,
            )
        if verbose: print('\nOutput layer trained.\n')

if __name__=='__main__':
    pass
