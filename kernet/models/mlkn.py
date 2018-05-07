# -*- coding: utf-8 -*-
# torch 0.3.1

from __future__ import print_function, division
import types

import torch
from torch.autograd import Variable

import sys
sys.path.append('../kernet')
import backend as K
from layers.kerlinear import kerLinear
from layers.ensemble import kerLinearEnsemble
# BUG: import is buggy if this script is run from kernet/kernet/models/

# TODO: multi-GPU support
# TODO: python2 compatibility


torch.manual_seed(1234)

class baseMLKN(torch.nn.Module):
    """
    Model for fast implementations of MLKN. Do not use this base class, use
    subclasses instead.
    """
    def __init__(self):
        super(baseMLKN, self).__init__()
        self._layer_counter = 0

    def add_layer(self, layer):
        """
        Add a layer to the model.

        Parameters
        ----------
        layer : a layer instance.
        """
        assert isinstance(layer, torch.nn.Module)
        setattr(self, 'layer'+str(self._layer_counter), layer)
        self._layer_counter += 1
        # layer indexing : layer 0 is closest to input

    def add_loss(self, loss_fn):
        """
        Specify loss function for the last layer. This is the objective function
        to be minimized by the optimizer.

        Parameters
        ----------
        loss_fn : callable, torch loss object
        """
        # CrossEntropyLoss (CrossEntropyLoss(output, y), in PyTorch, is
        # equivalent to NLLLoss(logsoftmax(output), y), where logsoftmax =
        # torch.nn.LogSoftmax(dim=1) for output Tensor of shape
        # (n_example, n_class)). Base of the log functions in these operations
        # is e.
        setattr(self, 'output_loss_fn', loss_fn)

    def add_metric(self, metric_fn):
        """
        Specify a metric against which the model is evaluated.

        Parameters
        ----------
        metric_fn : callable, torch loss object
        """
        setattr(self, '_metric_fn', metric_fn)

    def forward(self, x, upto=None, update_X=False):
        """
        Feedforward upto layer 'upto'. If 'upto' is not passed,
        this works as the standard forward function in PyTorch.

        Parameters
        ----------

        x : Tensor, shape (batch_size, dim)
            Batch.

        upto (optional) : int
            Index for the layer upto (and including) which we will evaluate
            the model. 0-indexed.

        Returns
        -------
        y : Tensor, shape (batch_size, out_dim)
            Batch output.
        """
        # NOTE: to use this method, make sure all layers in the model are
        # kerLinear
        if upto is not None: # cannot use 'if upto' here since it is 0-indexed
        # and layer0 is the first layer
            assert 0<=upto<=self._layer_counter
            counter = upto + 1
        else: counter = self._layer_counter

        y_previous = x
        for i in range(counter):

            layer = getattr(self, 'layer'+str(i))

            if i > 0 and update_X:
                if isinstance(layer.X, types.GeneratorType): # for ensemble layer
                    for j in range(layer._comp_counter):
                        comp = getattr(layer, 'comp'+str(j))
                        comp.X = self.forward(comp.X_init, upto=i-1)
                else:
                    layer.X = self.forward(layer.X_init, upto=i-1)
                    # NOTE: if do not update using X_init, shape[1] of this
                    # data will not match data in previous layers, will cause
                    # issue in kerMap
            y = layer(y_previous)
            y_previous = y

        return y

    def evaluate(self, X_test, Y_test=None, layer=None, batch_size=None, write_to=None):
        """
        Feed in X_test and get output at a specified layer.

        Parameters
        ----------
        X_test : Tensor, shape (n_example, dim)

        Y_test (optional) : Tensor, shape (n_example, y_dim)
            For calculating the metric value.

        layer (optional) : int
            Output of this layer is returned. Layers are
            zero-indexed with the 0th layer being the one closest to the input.
            If this parameter is not passed, evaluate the output of the entire
            network and calculate and print the metric value.

        Returns
        -------
        X_eval : Tensor, shape (n_example, layer_dim)
            Hidden representation of X_test at the given layer.
        """
        # TODO: test
        if not batch_size or batch_size>X_test.shape[0]: batch_size=X_test.shape[0]
        if layer is None: layer=self._layer_counter-1
        else: assert 0<=layer<=self._layer_counter-1

        layer, layer_index = getattr(self, 'layer'+str(layer)), layer
        out_dim = layer.out_dim

        X_eval = Variable(torch.cuda.FloatTensor(X_test.shape[0], out_dim),
        requires_grad=False) if \
        X_test.is_cuda else \
        Variable(torch.FloatTensor(X_test.shape[0], out_dim), requires_grad=False)

        i = 0
        for x_test in K.get_batch(X_test, batch_size=batch_size):
            x_test = x_test[0].clone() # NOTE: clone turns x_test into a leaf
            # Variable, which is required to set the volatile flag

            # NOTE: when only one set is sent to get_batch,
            # we need to use x_test[0] because no automatic unpacking has
            # been done by Python
            x_test.volatile = True
            x_eval = self.forward(
                x_test,
                upto=layer_index,
                update_X=False
                )

            if x_test.shape[0]<batch_size: # last batch
                X_eval[i*batch_size:] = x_eval.data[:]
                break
            X_eval[i*batch_size: (i+1)*batch_size] = x_eval.data[:]
            i += 1

        if layer_index == self._layer_counter-1 and Y_test is not None:
            # compute and print the metric if the entire model has been evaluated
            # only evaluate when Y_test is provided
            if self._metric_fn.__class__.__name__=='L0Loss':
                # predict first
                _, Y_pred = torch.max(X_eval, dim=1)
                print('{}: {:.3f}'.format(
                    'L0Loss (%)',
                    self._metric_fn(y_pred=Y_pred, y=Y_test)*100
                ))
                # TODO: a better way to record history?
                if write_to:
                    print('{}: {:.3f}'.format(
                        'L0Loss (%)',
                        self._metric_fn(y_pred=Y_pred, y=Y_test)*100
                    ), file=open(write_to,'a'))
            else:
                # assumes self._metric_fn is a torch loss object
                print('{}: {:.3f}'.format(
                    self._metric_fn.__class__.__name__,
                    self._metric_fn(X_eval, Y_test).data[0]
                ))

        return X_eval
        # NOTE: this is to make the type of Y_pred consistent with X_test since
        # X_test must be a Variable

    def fit(self):
        raise NotImplementedError('must be implemented by subclass')

    def save(self):
        pass
        # TODO: wrap native PyTorch support for save

class MLKN(baseMLKN):
    """
    A general MLKN model that does everything. Trained using backpropagation.
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
        ):
        """
        Parameters
        ----------
        n_epoch : int
            The number of epochs to train the model.

        X : Tensor, shape (n_example, dim)
            Training set.

        Y : Tensor, shape (n_example, 1) or (n_example,)
            Target data.

        X_val (optional) : Tensor, shape (n_example, dim)
            Optional validation set.

        Y_val (optional) : Tensor, shape (n_example, 1) or (n_example,)

        val_window (optional) : int
            The number of epochs between validations.

        batch_size (optional) : int
            If not specified, use full mode.

        shuffle (optional) : bool
            Shuffle the data at each epoch.

        accumulate_grad (optional) : bool
            If True, accumulate gradient from each batch and only update the
            weights after each epoch.
        """
        assert X.shape[0]==Y.shape[0]

        if not batch_size or batch_size>X.shape[0]: batch_size = X.shape[0]
        n_batch = X.shape[0] // batch_size
        last_batch = bool(X.shape[0] % batch_size)

        if len(Y.shape)==2: Y=Y.view(-1,)
        # NOTE: CrossEntropyLoss requires label tensor to be of
        # shape (n)
        # TODO: what about multi-D MSELoss or CrossEntropyLoss?
        if isinstance(self.output_loss_fn, torch.nn.CrossEntropyLoss):
            Y=Y.type(torch.cuda.LongTensor)\
            if Y.is_cuda else Y.type(torch.LongTensor)
            # NOTE: required by CrossEntropyLoss

        elif isinstance(self.output_loss_fn, torch.nn.MSELoss):
            Y=Y.type(torch.cuda.FloatTensor)\
            if Y.is_cuda else Y.type(torch.FloatTensor)
            # NOTE: required by MSELoss

        for param in self.parameters(): param.requires_grad=True # unfreeze
        for _ in range(n_epoch):
            __ = 0
            self.optimizer.zero_grad()

            for x, y in K.get_batch(
                X, Y,
                batch_size=batch_size,
                shuffle=shuffle
                ):
                update_X = True if _==0 and __==0 else False

                __ += 1
                output = self.forward(x, update_X=update_X)

                loss = self.output_loss_fn(output, y)
                # NOTE: L2 regulatization
                # is taken care of by setting the weight_decay param in the
                # optimizer, see
                # https://discuss.pytorch.org/t/simple-l2-regularization/139

                print('epoch: {}/{}, batch: {}/{}, loss({}): {:.3f}'.format(
                    _+1, n_epoch, __, n_batch+int(last_batch),
                    self.output_loss_fn.__class__.__name__,
                    loss.data[0]
                    ))

                if not accumulate_grad:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.forward(x, update_X=True) # update X after each step

                else:
                    loss.backward(retain_graph=True)
            if accumulate_grad:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.forward(x, update_X=True)

            if X_val is not None and (_+1) % val_window==0:
                self.evaluate(X_test=X_val, Y_test=Y_val)

        print('\n' + '#'*10 + '\n')
        for param in self.parameters(): param.requires_grad=False # freeze
        # the model

class MLKNGreedy(baseMLKN):
    """
    Base model for a greedy MLKN. Do not use this class, use subclass instead.
    """
    # TODO: this model uses kerLinear or nn.Linear (build thickLinear to make
    # kernel-neural hybrid simpler (note: no kernel should be closer to input
    # than neural)) layers only and all layers
    # should be trained as RBFNs. Users should be encouraged to use the
    # layerwise functionality of this model. Build another model that allows
    # passing in sklearn.SVM objects in order to train the last layer as a SVM
    def __init__(self):
        super(MLKNGreedy, self).__init__()
        self._optimizer_counter = 0

    def add_optimizer(self, optimizer):
        """
        Configure layerwise training. One needs to designate an optimizer for
        each layer. The ordering for optimizers follows that of the layers.
        User can pass any iterable to the non-optional 'params' parameter
        of the optimizer object and this will be later overwritten by this
        function.
        """
        assert isinstance(optimizer, torch.optim.Optimizer)
        setattr(self, 'optimizer'+str(self._optimizer_counter), optimizer)
        self._optimizer_counter += 1
        # optimizer indexing : optimizer 0 is the optimizer for layer 0

    def _compile(self):
        """
        Compile the model.
        """
        assert self._optimizer_counter==self._layer_counter
        # assign each optimizer to its layer ###################################
        for i in range(self._optimizer_counter):
            layer = getattr(self, 'layer'+str(i))
            optimizer = getattr(self, 'optimizer'+str(i))
            optimizer.param_groups[0]['params'] = list(layer.parameters())

        for param in self.parameters(): param.requires_grad=False # Freeze all
        # layers. Only unfreeze a layer when optimizing it, then the amount of
        # calculation for gradients is minimized for each step.

    def _fit_hidden(
        self,
        n_epoch,
        X, Y,
        n_group,
        batch_size=None,
        shuffle=False,
        accumulate_grad=True,
        keep_grad=False):
        """
        Fit the representation learning layers, i.e., all layers but the last.
        """

        assert len(Y.shape) <= 2
        # NOTE: this model only supports hard class labels
        assert X.shape[0]==Y.shape[0]

        if not batch_size or batch_size>X.shape[0]: batch_size = X.shape[0]
        n_batch = X.shape[0] // batch_size
        last_batch = bool(X.shape[0] % batch_size)

        # train the representation-learning layers #############################
        if len(Y.shape)==1: Y=Y.view(-1, 1)
        # NOTE: ideal_gram() requires label tensor to be of shape
        # (n, 1)
        loss_fn = torch.nn.CosineSimilarity() # NOTE: equivalent to alignment
        for i in range(self._layer_counter-1):
            optimizer = getattr(self, 'optimizer'+str(i))
            next_layer = getattr(self, 'layer'+str(i+1))
            layer = getattr(self, 'layer'+str(i))
            # assert isinstance(next_layer, (kerLinear, kerLinearEnsemble)) # TODO
            # NOTE:
            # torch.nn.Linear cannot pass this. We do this check because each
            # layer uses the kernel function from the next layer to calculate
            # loss but nn.Linear does not have a kernel so it cannot be the
            # next layer for any layer

            for param in layer.parameters(): param.requires_grad=True # unfreeze

            for _ in range(n_epoch[i]):
                __ = 0
                optimizer.zero_grad()
                for x, y in K.get_batch(
                    X, Y,
                    batch_size=batch_size,
                    shuffle=shuffle
                    ):
                    update_X = True if _==0 and __==0 else False
                    __ += 1
                    # get ideal gram matrix ####################################
                    ideal_gram = K.ideal_gram(y, y, n_group)
                    ideal_gram=ideal_gram.type(torch.cuda.FloatTensor)\
                    if ideal_gram.is_cuda else ideal_gram.type(torch.FloatTensor)
                    # NOTE: required by CosineSimilarity

                    # get output ###############################################
                    output = self.forward(x, upto=i, update_X=update_X)
                    # output.register_hook(print)
                    # print('output', output) # NOTE: layer0 initial feedforward
                    # passed

                    gram = K.kerMap(
                        output,
                        output,
                        next_layer.sigma
                        )
                    # print(gram) # NOTE: initial feedforward passed
                    # gram.register_hook(print) # NOTE: (for torch_backend.alignment)
                    # gradient here is inconsistent using the alignment loss from
                    # torch_backend: hand-calculated_grad*n_example =
                    # pytorch_grad

                    # compute loss and optimizer takes a step###################
                    loss = -loss_fn(gram.view(1, -1), ideal_gram.view(1, -1))
                    # NOTE: negative alignment
                    # NOTE: L2 regulatization
                    # is taken care of by setting the weight_decay param in the
                    # optimizer, see
                    # https://discuss.pytorch.org/t/simple-l2-regularization/139

                    print('epoch: {}/{}, batch: {}/{}, loss({}): {:.3f}'.format(
                        _+1, n_epoch[i], __, n_batch+int(last_batch),
                        'Alignment',
                        -loss.data[0]
                        ))

                    loss.backward()
                    # train the layer
                    if not accumulate_grad:
                        optimizer.step()
                        if not keep_grad:
                            optimizer.zero_grad()

                if accumulate_grad:
                    optimizer.step()
                    if not keep_grad:
                        optimizer.zero_grad()

            print('\n' + '#'*10 + '\n')
            for param in layer.parameters(): param.requires_grad=False # freeze
            # this layer again

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
        keep_grad=False
        ):
        """
        Fit the last layer.
        """

        assert len(Y.shape) <= 2 # NOTE: this model only supports hard class labels
        assert X.shape[0]==Y.shape[0]

        if not batch_size or batch_size>X.shape[0]: batch_size = X.shape[0]
        n_batch = X.shape[0] // batch_size
        last_batch = bool(X.shape[0] % batch_size)


        # train the last layer as a RBFN classifier ############################
        i = self._layer_counter-1
        optimizer = getattr(self, 'optimizer'+str(i))
        layer = getattr(self, 'layer'+str(i))

        if len(Y.shape)==2: Y=Y.view(-1,)
        # NOTE: CrossEntropyLoss (and probably also MSELoss) requires label
        # tensor to be of shape (n)

        # TODO: what about multi-D MSELoss or CrossEntropyLoss?

        if isinstance(self.output_loss_fn, torch.nn.CrossEntropyLoss):
            Y=Y.type(torch.cuda.LongTensor)\
            if Y.is_cuda else Y.type(torch.LongTensor)
            # NOTE: required by CrossEntropyLoss

        elif isinstance(self.output_loss_fn, torch.nn.MSELoss):
            Y=Y.type(torch.cuda.FloatTensor)\
            if Y.is_cuda else Y.type(torch.FloatTensor)
            # NOTE: required by MSELoss


        for param in layer.parameters(): param.requires_grad=True # unfreeze
        for _ in range(n_epoch[i]):
            __ = 0
            optimizer.zero_grad()
            for x, y in K.get_batch(
                X, Y,
                batch_size=batch_size,
                shuffle=shuffle
                ):
                update_X = True if _==0 and __==0 else False
                __ += 1
                # compute loss
                output = self.forward(x, upto=i, update_X=update_X)
                # print(output) # NOTE: layer1 initial feedforward passed

                loss = self.output_loss_fn(output, y)
                # print(loss) # NOTE: initial feedforward passed
                # NOTE: L2 regulatization
                # is taken care of by setting the weight_decay param in the
                # optimizer, see
                # https://discuss.pytorch.org/t/simple-l2-regularization/139

                print('epoch: {}/{}, batch: {}/{}, loss({}): {:.3f}'.format(
                    _+1, n_epoch[i], __, n_batch+int(last_batch),
                    self.output_loss_fn.__class__.__name__,
                    loss.data[0]
                    ))

                loss.backward()
                # train the layer
                if not accumulate_grad:
                    optimizer.step()
                    if not keep_grad:
                        optimizer.zero_grad()

                #########
                # define crossentropy loss to test gradient
                # loss = output_prob.mul(K.one_hot(y.unsqueeze(dim=1), n_class)).sum()/2
                # NOTE: this calculation results in the same gradient as that
                # calculated by autograd using CrossEntropyLoss as loss_fn
                #########
            if accumulate_grad:
                optimizer.step()
                if not keep_grad:
                    optimizer.zero_grad()

            if X_val is not None and (_+1) % val_window==0:
                self.evaluate(X_test=X_val, Y_test=Y_val, write_to=write_to)

        print('\n' + '#'*10 + '\n')
        for param in layer.parameters(): param.requires_grad=False # freeze
        # this layer again

class MLKNClassifier(MLKNGreedy):
    """
    MLKN classifier dedicated for layerwise training using method proposed
    in https://arxiv.org/abs/1802.03774. This feature can be tedious to
    implement in standard PyTorch since a lot of details need to be taken care
    of so we build it for your convenience.

    If one wants a MLKN classifier trained with standard backpropagation,
    use MLKN instead, the setup for training would be much simpler for
    MLKN and many more loss functions are supported.
    """
    def __init__(self):
        super(MLKNClassifier, self).__init__()

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
        write_to=None,
        keep_grad=False
        ):
        """
        Parameters
        ----------
        n_epoch : tuple
            The number of epochs for each layer. If the length of the tuple is
            greater than the number of the layers, use n_epoch[:n_layer].
            Even if there is only one layer, this parameter must be a tuple (
            may be of of a scalar, e.g., (1,)).

        X : Tensor, shape (n_example, dim)
            Training set.

        Y : Variable of shape (n_example, 1) or (n_example,)
            Categorical labels for the set.

        n_class : int

        X_val (optional) : Tensor, shape (n_example, dim)
            Optional validation set.

        Y_val (optional) : Tensor, shape (n_example, 1) or (n_example,)

        val_window (optional) : int
            The number of epochs between validations.

        write_to (optional) : str
            File to record hyperparameters and write a history of validation
            metric to. Default write mode
            is append, so make sure to create a new file for each run.

        batch_size (optional) : int
            If not specified, use full mode.

        shuffle (optional) : bool
            Shuffle the data at each epoch.

        accumulate_grad (optional) : bool
            If True, accumulate gradient from each batch and only update the
            weights after each epoch.

        keep_grad (optional) : bool
            If True, will not zero grad after each grad update (which will lead
            to wrong results!). Set to True for internal testing only.
        """
        # TODO
        """
        if write_to is not None:

            # TODO: write settings to a file, only supports kerLinear and
            # kerLinearEnsemble objects
            # TODO: include more info such as loss functions, optimizers, etc.
            model = {}
            for i in range(self._layer_counter):
                layer = getattr(self, 'layer'+str(i))
                model['layer'+str(i)] = \
                    ('sigma', layer.sigma, 'out_dim', layer.out_dim)]

            settings = {'n_epoch':n_epoch, 'model':config, 'lr':()
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
            keep_grad=keep_grad
            )
        print('Hidden layers trained.')

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
            keep_grad=keep_grad
            )
        print('Output layer trained.')

if __name__=='__main__':
    pass
