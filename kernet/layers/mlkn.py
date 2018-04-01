# -*- coding: utf-8 -*-
# torch 0.3.1

import torch
from torch.autograd import Variable
import torch_backend as K # TODO: relative import
from kerlinear import kerLinear

# TODO: check GPU compatibility: move data and modules on GPU, see, for example,
# https://github.com/pytorch/pytorch/issues/584

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

class MLKNClassifier(baseMLKN):
    """
    MLKN classifier dedicated for layerwise training using method proposed
    in https://arxiv.org/abs/1802.03774. This feature can be tedious to
    implement in standard PyTorch since a lot of details need to be taken care
    of.
    
    If one wants a MLKN classifier trained with standard backpropagation,
    use MLKNGeneral instead, the setup for training would be much simpler for
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

    def predict(self, x, X):
        """
        Get predictions from the classifier.

        Parameters
        ----------

        x : Tensor, shape (batch_size, dim)
            Batch. Must be a leaf Variable.

        X : Tensor, shape (n_example, dim)
            Training set.

        Returns
        -------
        y_pred : Tensor, shape (batch_size,)
            Predicted labels for the batch.
        """
        y_raw = self.forward_volatile(x, X)
        _, y_pred = torch.max(y_raw, dim=1)
        return y_pred

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

    def fit(self, n_epoch, batch_size, x, X, y, n_class):
        """
        Parameters
        ----------

        x : Tensor, shape (batch_size, dim)
            Batch.

        X : Tensor, shape (n_example, dim)
            Training set.

        y : Variable of shape (n_example, 1) or (n_example,)
            Categorical labels for the batch.
        """
        assert self._optimizer_counter==self._layer_counter

        # assign each optimizer to its layer ###################################
        for i in range(self._optimizer_counter):
            layer = getattr(self, 'layer'+str(i))
            """
            #########
            # set the weights to some value to test if the network gives
            # the same results as those calculated by hand, this test uses
            # a two-layer network
            if i==0:
                layer.weight.data = torch.FloatTensor([[.1, .2], [.5, .7]])
                layer.bias.data = torch.FloatTensor([0, 0])
            if i==1:
                layer.weight.data = torch.FloatTensor([[1.2, .3], [.2, 1.7]])
                layer.bias.data = torch.FloatTensor([0.1, 0.2])

            #########
            """


            optimizer = getattr(self, 'optimizer'+str(i))
            optimizer.param_groups[0]['params'] = list(layer.parameters())

        for param in self.parameters(): param.requires_grad=False # freeze all
        # layers

        # train the representation-learning layers #############################
        assert len(y.shape) <= 2
        if len(y.shape)==1: y.unsqueeze_(dim=1)
        # NOTE: ideal_gram requires label tensor to be of shape (n, 1)

        ideal_gram = K.ideal_gram(y, y, n_class)
        ideal_gram=ideal_gram.type(torch.cuda.FloatTensor)\
        if ideal_gram.is_cuda else ideal_gram.type(torch.FloatTensor)
        # NOTE: required by MSELoss

        # loss_fn = torch.nn.MSELoss()
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
                # compute loss
                output = self.forward(x, X, upto=i)
                # output.register_hook(print) # NOTE: (for alignment)
                # see note, there is a problem here
                #########
                # print('output', output) # NOTE: layer0 initial feedforward passed
                #########
                gram = K.kerMap(
                    output,
                    output,
                    next_layer.sigma
                    )
                #########
                # print(gram) # NOTE: initial feedforward passed
                #########
                # gram.register_hook(print) # NOTE: (for alignment)
                # gradient here is off by *0.5: hand-calculated grad * 2 =
                # pytorch grad

                # BUG: using alignment loss causes gradient problem, see
                # comments from past commits
                """
                alignment = K.alignment(ideal_gram, gram)
                # print(alignment) # NOTE: initial feedforward passed
                loss = -alignment
                """

                # loss = loss_fn(gram, ideal_gram)

                loss = -loss_fn(gram.view(1, -1), ideal_gram.view(1, -1))
                # NOTE: negative alignment
                # NOTE: L2 regulatization
                # is taken care of by setting the weight_decay param in the
                # optimizer, see
                # https://discuss.pytorch.org/t/simple-l2-regularization/139

                print(_, -loss.data[0])

                # train the layer
                optimizer.zero_grad()
                loss.backward()

                #########
                # check gradient
                # print('weight', layer.weight)
                # print('gradient', layer.weight.grad.data) # NOTE: see note
                #########

                optimizer.step()

            for param in layer.parameters(): param.requires_grad=False # freeze
            # this layer again



        # train the last layer as a RBFN classifier ############################
        i = self._layer_counter-1
        optimizer = getattr(self, 'optimizer'+str(i))
        layer = getattr(self, 'layer'+str(i))

        if len(y.shape)==2: y.squeeze_(dim=1)
        # NOTE: CrossEntropyLoss requires label tensor to be of
        # shape (n)

        y=y.type(torch.cuda.LongTensor)\
        if y.is_cuda else y.type(torch.LongTensor)
        # NOTE: required by CrossEntropyLoss

        for param in layer.parameters(): param.requires_grad=True # unfreeze
        for _ in range(n_epoch[i]):
            # compute loss
            output = self.forward(x, X, upto=i)
            # print(output) # NOTE: layer1 initial feedforward passed

            loss = self.output_loss_fn(output, y)
            # print(loss) # NOTE: initial feedforward passed
            # NOTE: L2 regulatization
            # is taken care of by setting the weight_decay param in the
            # optimizer, see
            # https://discuss.pytorch.org/t/simple-l2-regularization/139

            print(_, loss.data[0])

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
        for param in layer.parameters(): param.requires_grad=False # freeze
        # this layer again

if __name__=='__main__':
    pass
