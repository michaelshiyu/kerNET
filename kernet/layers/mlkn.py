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
        assert self._optimizer_counter==self._layer_counter
        for i in range(self._optimizer_counter):
            layer = getattr(self, 'layer'+str(i))
            optimizer = getattr(self, 'optimizer'+str(i))
            optimizer.param_groups[0]['params'] = list(layer.parameters())
            # TODO: why is optimizer.param_groups a list containing a dict
            # rather than a dict?

        # TODO: get ideal before going into the loop
        ideal_gram = K.ideal_gram(y, y, n_class)
        for i in range(self._layer_counter-1):
            # train the representation-learning layers
            optimizer = getattr(self, 'optimizer'+str(i))
            next_layer = getattr(self, 'layer'+str(i+1))
            assert isinstance(next_layer, kerLinear) # TODO: see if
            # torch.nn.Linear can pass this and ban it if it can

            for _ in range(n_epoch[i]):
                # compute loss
                output = self.forward(x, X, upto=i)
                gram = K.kerMap(
                    output,
                    output,
                    next_layer.sigma
                    )

                alignment = K.alignment(ideal_gram, gram)
                loss = -alignment # TODO: add regularization terms
                # mse = torch.nn.MSELoss(size_average=False)
                # loss = mse(ideal_gram, gram)

                print(_, loss.data[0])

                # train the layer
                optimizer.zero_grad()
                loss.backward() # TODO: freeze some subgraphs as only one layer
                # needs to be differentiated at a time
                optimizer.step()


        # TODO: train the last layer as a RBFN classifier,  allow specifying
        # loss used




if __name__=='__main__':
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor

    x = Variable(torch.FloatTensor([[0, 7], [1, 2]]).type(dtype), requires_grad=False)
    X = Variable(torch.FloatTensor([[1, 2], [3, 4], [5, 6]]).type(dtype), requires_grad=False)
    y = Variable(torch.FloatTensor([[0], [1], [1]]).type(dtype), requires_grad=False)

    mlkn = MLKNClassifier()
    mlkn.add_layer(kerLinear(ker_dim=X.shape[0], out_dim=5, sigma=1, bias=True))
    mlkn.add_layer(kerLinear(ker_dim=X.shape[0], out_dim=2, sigma=1, bias=True))

    mlkn.add_optimizer(torch.optim.SGD(params=mlkn.parameters(), lr=1e-1))
    mlkn.add_optimizer(torch.optim.SGD(params=mlkn.parameters(), lr=1e-1))

    mlkn.fit(n_epoch=(10000, 10), batch_size=50, x=X, X=X, reg_coef=.1, y=y, n_class=2)

    """
    y_pred = mlkn(x=x, X=X)
    print('y_pred', y_pred)
    print('weight', mlkn.layer0.linear.weight)
    print('bias', mlkn.layer0.linear.bias)
    print('params', list(mlkn.parameters()))


    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(mlkn.parameters(), lr=1e-4)
    for t in range(500):
        y_pred = mlkn(x, X)
        loss = criterion(y_pred, y)
        print(t, loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    """
