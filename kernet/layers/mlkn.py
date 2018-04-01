# -*- coding: utf-8 -*-
# torch 0.3.1
from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch_backend as K # TODO: relative import
from kerlinear import kerLinear

# TODO: check GPU compatibility: move data and modules on GPU, see, for example,
# https://github.com/pytorch/pytorch/issues/584
# TODO: using multiple devices, see
# http://pytorch.org/docs/0.3.1/notes/multiprocessing.html and nn.DataParallel
# TODO: check numerical grad for the toy example

# TODO: relative import; tests

torch.manual_seed(1234)

class baseMLKN(torch.nn.Module):
    """
    Model for fast implementations of MLKN. Do not use this base class, use
    subclasses instead.

    A special property of MLKN is that for it to do anything, it always needs
    a reference to its training set, which we call X. You could think of this
    as a set of bases to expand your kernel machine on. And this is why in a lot
    of methods of this class you see the parameter X. It is true that it
    can be highly memory-inefficient to carry this big chunk of data around,
    we may add more functionalities to the class in the future to tackle with
    this issue.
    """
    # TODO: see above documentation
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

    def forward(self, x, X, upto=None):
        """
        Feedforward upto layer 'upto'. If 'upto' is not passed,
        this works as the standard forward function in PyTorch.

        Parameters
        ----------

        x : Tensor, shape (batch_size, dim)
            Batch.

        X : Tensor, shape (n_example, dim)
            Training set.

        upto (optional) : int
            Index for the layer upto (and including) which we will evaluate
            the model. 0-indexed.

        Returns
        -------
        y : Tensor, shape (batch_size, out_dim)
            Batch output.
        """
        if upto is not None: # cannot use 'if upto' here since it is 0-indexed
        # and layer0 is the first layer
            assert 0<=upto<=self._layer_counter
            counter = upto + 1
        else: counter = self._layer_counter

        y_previous, Y_previous = x, X
        # TODO: because we always need to compute F_i(X) at each layer i, this
        # is a huge overhead
        # feedforward
        for i in range(counter):
            layer = getattr(self, 'layer'+str(i))
            y, Y = layer(y_previous, Y_previous), layer(Y_previous, Y_previous)
            y_previous, Y_previous = y, Y

        return y

    def forward_volatile(self, x, X, upto=None):
        """
        Feedforward upto layer 'upto' in volatile mode. Use for inference. See
        http://pytorch.org/docs/0.3.1/notes/autograd.html.
        If 'upto' is not passed, this works as the standard forward function
        in PyTorch.

        Parameters
        ----------

        x : Tensor, shape (batch_size, dim)
            Batch. Must be a leaf Variable.

        X : Tensor, shape (n_example, dim)
            Training set.
        upto (optional) : int
            Index for the layer upto (and including) which we will evaluate
            the model. 0-indexed.

        Returns
        -------
        y : Tensor, shape (batch_size, out_dim)
            Batch output.
        """
        x.volatile = True
        return self.forward(x, X, upto)

    def fit(self):
        raise NotImplementedError('must be implemented by subclass')

    def save(self):
        pass
        # TODO: wrap native PyTorch support for save

class MLKN(baseMLKN):
    """
    A general MLKN model that does everything. Trained using backpropagation.
    """
    # NOTE: maybe this could be absorbed into baseMLKN
    def add_loss(self,):
        pass
    def add_optimizer(self,):
        pass
    def fit(self,):
        """BackProp."""
        pass

class MLKNClassifier(baseMLKN):
    """
    MLKN classifier dedicated for layerwise training using method proposed
    in https://arxiv.org/abs/1802.03774. This feature can be tedious to
    implement in standard PyTorch since a lot of details need to be taken care
    of so we build it for your convenience.

    If one wants a MLKN classifier trained with standard backpropagation,
    use MLKN instead, the setup for training would be much simpler for
    MLKNGeneral and many more loss functions are supported.
    """
    # TODO: this model uses kerLinear or nn.Linear (build thickLinear to make
    # kernel-neural hybrid simpler (note: no kernel should be closer to input
    # than neural)) layers only and all layers
    # should be trained as RBFNs. Users should be encouraged to use the
    # layerwise functionality of this model. Build another model that allows
    # passing in sklearn.SVM objects in order to train the last layer as a SVM
    def __init__(self):
        super(MLKNClassifier, self).__init__()
        self._optimizer_counter = 0

    def add_loss(self, loss_fn):
        """
        Specify loss function for the last layer. We recommend using
        CrossEntropyLoss (CrossEntropyLoss(output, y), in PyTorch, is
        equivalent to NLLLoss(logsoftmax(output), y), where logsoftmax =
        torch.nn.LogSoftmax(dim=1) for output Tensor of shape
        (n_example, n_class)). Base of the log functions in these operations
        is e.

        Using other loss functions may cause unexpected behaviors since
        we have only tested the model with CrossEntropyLoss.

        Parameters
        ----------
        loss_fn : a torch loss object
        """
        setattr(self, 'output_loss_fn', loss_fn)

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

    def predict(self, X_test, X, batch_size=None):
        """
        Get predictions from the classifier.

        Parameters
        ----------

        X_test : Tensor, shape (n1_example, dim)
            Test set.

        X : Tensor, shape (n_example, dim)
            Training set.

        Returns
        -------
        Y_pred : Tensor, shape (batch_size,)
            Predicted labels.
        """
        if not batch_size: batch_size=X_test.shape[0]
        Y_pred = torch.cuda.LongTensor(X_test.shape[0],) if X_test.is_cuda\
        else torch.LongTensor(X_test.shape[0],)
        i = 0
        for x_test in K.get_batch(X_test, batch_size=batch_size):
            x_test = x_test[0].clone() # NOTE: clone turns x_test into a leaf
            # Variable, which is required to set the volatile flag

            # TODO: figure out why when only one set is sent to get_batch,
            # we need to use x_test[0] but not directly x_test
            y_raw = self.forward_volatile(x_test, X)

            _, y_pred = torch.max(y_raw, dim=1)

            if x_test.shape[0]<batch_size: # last batch
                Y_pred[i*batch_size:] = y_pred.data[:]
                break
            Y_pred[i*batch_size: (i+1)*batch_size] = y_pred.data[:]
            i += 1

        return Variable(Y_pred, requires_grad=False)
        # NOTE: this is to make the type of Y_pred consistent with X_test since
        # X_test must be a Variable

    def get_error(self, y_pred, y):
        """
        Compute prediction error rate.

        Parameters
        ----------

        y_pred : Tensor, shape (batch_size,)
            Predicted labels.

        y : Tensor, shape (batch_size,)
            True labels.

        Returns
        -------
        err : scalar (or wrapped in a Variable, if one of y or y_pred is)
            Error rate.
        """
        assert y_pred.shape==y.shape
        y_pred = y_pred.type_as(y)
        err = (y_pred!=y).sum().type(torch.FloatTensor).div_(y.shape[0])
        return err

    def fit(self, n_epoch, X, Y, n_class, batch_size=None, shuffle=False):
        """
        Parameters
        ----------
        n_epoch : tuple
            The number of epochs for each layer. If the length of the tuple is
            greater than the number of the layers, use n_epoch[:n_layer].

        X : Tensor, shape (n_example, dim)
            Training set.

        Y : Variable of shape (n_example, 1) or (n_example,)
            Categorical labels for the set.

        batch_size (optional) : int
            If not specified, use full mode.

        shuffle (optional) : bool
            Shuffle the data at each epoch.
        """
        # TODO: now uses SGD or batch GD: update weights and empty grad per batch.
        # Give option on using full GD: accumulate grad from each batch and
        # only update once per epoch

        assert self._optimizer_counter==self._layer_counter
        assert len(Y.shape) <= 2 # NOTE: this model only supports hard class labels
        assert X.shape[0]==Y.shape[0]

        if not batch_size: batch_size = X.shape[0]
        n_batch = X.shape[0] // batch_size
        last_batch = bool(X.shape[0] % batch_size)

        # assign each optimizer to its layer ###################################
        for i in range(self._optimizer_counter):
            layer = getattr(self, 'layer'+str(i))
            optimizer = getattr(self, 'optimizer'+str(i))
            optimizer.param_groups[0]['params'] = list(layer.parameters())

        for param in self.parameters(): param.requires_grad=False # freeze all
        # layers

        # train the representation-learning layers #############################
        if len(Y.shape)==1: Y=Y.view(-1, 1)
        # NOTE: ideal_gram() requires label tensor to be of shape
        # (n, 1)
        loss_fn = torch.nn.CosineSimilarity() # NOTE: equivalent to alignment
        for i in range(self._layer_counter-1):
            optimizer = getattr(self, 'optimizer'+str(i))
            next_layer = getattr(self, 'layer'+str(i+1))
            layer = getattr(self, 'layer'+str(i))

            assert isinstance(next_layer, kerLinear)
            # NOTE:
            # torch.nn.Linear cannot pass this. We do this check because each
            # layer uses the kernel function from the next layer to calculate
            # loss but nn.Linear does not have a kernel so it cannot be the
            # next layer for any layer

            for param in layer.parameters(): param.requires_grad=True # unfreeze

            for _ in range(n_epoch[i]):
                __ = 0
                for x, y in K.get_batch(
                    X, Y,
                    batch_size=batch_size,
                    shuffle=shuffle
                    ):
                    __ += 1
                    # get ideal gram matrix ####################################
                    ideal_gram = K.ideal_gram(y, y, n_class)
                    ideal_gram=ideal_gram.type(torch.cuda.FloatTensor)\
                    if ideal_gram.is_cuda else ideal_gram.type(torch.FloatTensor)
                    # NOTE: required by CosineSimilarity

                    # get output ###############################################
                    output = self.forward(x, X, upto=i)
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

                    # train the layer
                    optimizer.zero_grad()
                    loss.backward()

                    #########
                    # check gradient
                    # print('weight', layer.weight)
                    # print('gradient', layer.weight.grad.data)
                    #########

                    optimizer.step()
            print('\n' + '#'*10 + '\n')
            for param in layer.parameters(): param.requires_grad=False # freeze
            # this layer again

        # train the last layer as a RBFN classifier ############################
        i = self._layer_counter-1
        optimizer = getattr(self, 'optimizer'+str(i))
        layer = getattr(self, 'layer'+str(i))

        if len(Y.shape)==2: Y=Y.view(-1,)
        # NOTE: CrossEntropyLoss requires label tensor to be of
        # shape (n)

        Y=Y.type(torch.cuda.LongTensor)\
        if Y.is_cuda else Y.type(torch.LongTensor)
        # NOTE: required by CrossEntropyLoss

        for param in layer.parameters(): param.requires_grad=True # unfreeze
        for _ in range(n_epoch[i]):
            __ = 0
            for x, y in K.get_batch(
                X, Y,
                batch_size=batch_size,
                shuffle=shuffle
                ):
                __ += 1
                # compute loss
                output = self.forward(x, X, upto=i)
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

                # train the layer
                optimizer.zero_grad()
                loss.backward()

                #########
                # define crossentropy loss to test gradient
                # loss = output_prob.mul(K.one_hot(y.unsqueeze(dim=1), n_class)).sum()/2
                # NOTE: this calculation results in the same gradient as that
                # calculated by autograd using CrossEntropyLoss as loss_fn

                # check gradient
                # print('weight', layer.weight)
                # print('gradient', layer.weight.grad.data)
                # print('bias gradient', layer.bias.grad.data)
                #########

                optimizer.step()
        print('\n' + '#'*10 + '\n')
        for param in layer.parameters(): param.requires_grad=False # freeze
        # this layer again

if __name__=='__main__':
    pass
