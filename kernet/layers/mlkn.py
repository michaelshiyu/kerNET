# -*- coding: utf-8 -*-
# torch 0.3.1

import torch
from torch.autograd import Variable
import torch_backend as K # TODO: relative import
from kerlinear import kerLinear

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

        X : Tensor, shape (n_example, dim)

        upto (optional) : int
            Index for the layer upto (and including) which we will evaluate
            the model. 0-indexed.

        Returns
        -------
        y : Tensor, shape (batch_size, out_dim)
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

    def fit(self):
        raise NotImplementedError('must be implemented by subclass')

class MLKNClassifier(baseMLKN):
    """
    MLKN classifier dedicated for layerwise training using method proposed
    in https://arxiv.org/abs/1802.03774. This feature can be tedious to
    implement in standard PyTorch since a lot of details need to be taken care
    of. If one wants a MLKN classifier trained with standard backpropagation,
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

    def fit(self, n_epoch, reg_coef, batch_size, x, X, y, n_class):
        """
        y : Variable of shape (n_example, 1)
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
            # TODO: why is optimizer.param_groups a list containing a dict
            # rather than a dict?

        for param in self.parameters(): param.requires_grad=False # freeze all
        # layers

        # train the representation-learning layers #############################
        ideal_gram = K.ideal_gram(y, y, n_class)
        for i in range(self._layer_counter-1):
            optimizer = getattr(self, 'optimizer'+str(i))
            next_layer = getattr(self, 'layer'+str(i+1))
            layer = getattr(self, 'layer'+str(i))
            assert isinstance(next_layer, kerLinear) # TODO: see if
            # torch.nn.Linear can pass this and ban it if it can because each
            # layer uses the kernel function from the next layer to calculate
            # loss but nn.Linear does not have a kernel so it cannot be the
            # next layer for any layer

            for param in layer.parameters(): param.requires_grad=True # unfreeze
            for _ in range(n_epoch[i]):
                # compute loss
                output = self.forward(x, X, upto=i)
                # output.register_hook(print) # NOTE: see note, there is a
                # problem here
                # print('output', output) # NOTE: layer0 initial feedforward passed
                gram = K.kerMap(
                    output,
                    output,
                    next_layer.sigma
                    )
                # print(gram) # NOTE: initial feedforward passed

                # gram.register_hook(print) # NOTE: gradient here is off by *0.5:
                # hand-calculated grad * 2 = pytorch grad

                alignment = K.alignment(ideal_gram, gram)
                # print(alignment) # NOTE: initial feedforward passed
                loss = -alignment # TODO: add regularization terms
                # mse = torch.nn.MSELoss(size_average=False)
                # loss = mse(ideal_gram, gram)

                print(_, loss.data[0])

                # train the layer
                optimizer.zero_grad()
                loss.backward()

                #########
                # check gradient
                # print('weight', layer.weight)
                # print('gradient', layer.weight.grad.data) # NOTE: see note
                # break
                #########

                # needs to be differentiated at a time
                optimizer.step()
            for param in layer.parameters(): param.requires_grad=False # freeze
            # this layer again



        # train the last layer as a RBFN classifier ############################
        i = self._layer_counter-1
        optimizer = getattr(self, 'optimizer'+str(i))
        layer = getattr(self, 'layer'+str(i))

        loss_fn = torch.nn.CrossEntropyLoss() # TODO: allow user specification
        # NOTE: CrossEntropyLoss combines softmax and crossentropy

        # loss_fn = torch.nn.NLLLoss() # NOTE: bases of the log used in NLLLoss and
        # CrossEntropyLoss are both e
        # logsoftmax = torch.nn.LogSoftmax(dim=1) # NOTE: softmax on dimension 1
        # for output Tensor of shape (n_example, n_class)

        y = y.type(torch.LongTensor) # TODO: support GPU

        assert len(y.shape) <= 2
        if len(y.shape)==2: y.squeeze_(dim=1)
        # NOTE: CrossEntropyLoss and NLLLoss require label tensor to be of
        # shape (n)

        for param in layer.parameters(): param.requires_grad=True # unfreeze
        for _ in range(n_epoch[i]):
            # compute loss
            output = self.forward(x, X, upto=i)
            # print(output) # NOTE: layer1 initial feedforward passed
            ######### NLLLoss
            # output_prob = logsoftmax(output) # log output probability
            # print(output_prob) # NOTE: initial feedforward passed
            # loss = loss_fn(output_prob, y)# TODO: add regularization terms
            # print(loss) # NOTE: initial feedforward passed
            #########

            ######### CrossEntropyLoss
            loss = loss_fn(output, y)
            # print(loss) # NOTE: initial feedforward passed
            #########

            # define crossentropy loss to test gradient
            # loss = output_prob.mul(K.one_hot(y.unsqueeze(dim=1), n_class)).sum()/2
            # NOTE: this calculation results in the same gradient as that
            # calculated by autograd

            print(_, loss.data[0])

            # train the layer
            optimizer.zero_grad()
            loss.backward()

            #########
            # check gradient
            # print('weight', layer.weight)
            # print('gradient', layer.weight.grad.data)
            # print('bias gradient', layer.bias.grad.data)
            # break
            #########

            optimizer.step()
        for param in layer.parameters(): param.requires_grad=False # freeze
        # this layer again

if __name__=='__main__':
    pass
