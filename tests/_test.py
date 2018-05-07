# -*- coding: utf-8 -*-
# torch 0.3.1

import unittest
import numpy as np
import torch
from torch.autograd import Variable

import sys
sys.path.append('../kernet/')
import backend as K
from models.mlkn import baseMLKN, MLKN, MLKNGreedy, MLKNClassifier
from layers.kerlinear import kerLinear
from layers.ensemble import kerLinearEnsemble


torch.manual_seed(1234)
np.random.seed(1234)

# forward test for baseMLKN done, bp fit does not need testing since it is
# simply a wrapper around native pytorch bp

# MLKNClassifier with kerLinear and kerLinearEnsemble
# tested forward and initial grad calculations on the toy example,
# does not test updated weights since updating the first layer would change
# grad of the second

# TODO: deeper models


# toy data
dtypeX, dtypeY = torch.FloatTensor, torch.LongTensor
if torch.cuda.is_available():
    dtypeX, dtypeY = torch.cuda.FloatTensor, torch.cuda.LongTensor
X = Variable(
    torch.FloatTensor([[1, 2], [3, 4]]).type(dtypeX),
    requires_grad=False
    )
Y = Variable(
    torch.FloatTensor([[0], [1]]).type(dtypeY),
    requires_grad=False
    )

#########
# base mlkn
mlkn = MLKNClassifier()
mlkn.add_layer(kerLinear(
    X=X,
    out_dim=2,
    sigma=3,
    bias=True
))
mlkn.add_layer(kerLinear(
    X=X,
    out_dim=2,
    sigma=2,
    bias=True
))
# manually set some weights
mlkn.layer0.weight.data = torch.FloatTensor([[.1, .2], [.5, .7]])
mlkn.layer0.bias.data = torch.FloatTensor([0, 0])
mlkn.layer1.weight.data = torch.FloatTensor([[1.2, .3], [.2, 1.7]])
mlkn.layer1.bias.data = torch.FloatTensor([0.1, 0.2])

#########
# ensemble

mlkn_ensemble = MLKNClassifier()
mlkn_ensemble.add_layer(K.to_ensemble(mlkn.layer0, batch_size=1))
mlkn_ensemble.add_layer(K.to_ensemble(mlkn.layer1, batch_size=1))

#########
# mlkn forward and evaluate
X_eval = mlkn(X, update_X=True)
X_eval_ = mlkn.evaluate(X)
# print(X_eval, X_eval_)

X_eval_hidden = mlkn(X, update_X=True, upto=0)
X_eval_hidden_ = mlkn.evaluate(X, layer=0)
# print(X_eval_hidden, X_eval_hidden_)

#########
# mlkn_ensemble forward and evaluate

X_eval = mlkn_ensemble(X, update_X=True)
X_eval_ = mlkn_ensemble.evaluate(X)
print(X_eval, X_eval_)

X_eval_hidden = mlkn_ensemble(X, update_X=True, upto=0)
X_eval_hidden_ = mlkn_ensemble.evaluate(X, layer=0)
print(X_eval_hidden, X_eval_hidden_)

#########
# mlkn weight update test

mlkn.add_optimizer(torch.optim.SGD(mlkn.parameters(), lr=0))
mlkn.add_optimizer(torch.optim.SGD(mlkn.parameters(), lr=0))

mlkn.add_loss(torch.nn.CrossEntropyLoss())
"""
mlkn.fit(
    n_epoch=(1, 1),
    X=X,
    Y=Y,
    n_class=2,
    keep_grad=True
    )

print(mlkn.layer0.weight.grad.data)
print(mlkn.layer0.bias.grad.data)
print(mlkn.layer1.weight.grad.data)
print(mlkn.layer1.bias.grad.data)
"""

#########
# mlkn_ensemble weight update test

mlkn_ensemble.add_optimizer(torch.optim.SGD(mlkn_ensemble.parameters(), lr=0))
mlkn_ensemble.add_optimizer(torch.optim.SGD(mlkn_ensemble.parameters(), lr=0))

mlkn_ensemble.add_loss(torch.nn.CrossEntropyLoss())
"""
mlkn_ensemble.fit(
    n_epoch=(1, 1),
    X=X,
    Y=Y,
    n_class=2,
    keep_grad=True
    )

for w in mlkn_ensemble.layer0.weight:
    print(w.grad.data)
for b in mlkn_ensemble.layer0.bias:
    if b is not None:
        print(b.grad.data)
for w in mlkn_ensemble.layer1.weight:
    print(w.grad.data)
for b in mlkn_ensemble.layer1.bias:
    if b is not None:
        print(b.grad.data)
"""
#########
# mlkn and mlkn_ensemble forward and evaluate (this test is nontrivial is lr>0
# (use it to test
# that params on mlkn_ensemble has been properly updated))

mlkn.optimizer0 = torch.optim.SGD(mlkn.parameters(), lr=0.5)
mlkn.optimizer1 = torch.optim.SGD(mlkn.parameters(), lr=0.5)

mlkn_ensemble.optimizer0 = torch.optim.SGD(mlkn_ensemble.parameters(), lr=0.5)
mlkn_ensemble.optimizer1 = torch.optim.SGD(mlkn_ensemble.parameters(), lr=0.5)

mlkn.fit(
    n_epoch=(10, 10),
    X=X,
    Y=Y,
    n_class=2,
    keep_grad=True
    )

mlkn_ensemble.fit(
    n_epoch=(10, 10),
    X=X,
    Y=Y,
    n_class=2,
    keep_grad=True
    )

X_eval = mlkn(X)
X_eval_ = mlkn.evaluate(X)
print(X_eval, X_eval_)

X_eval_hidden = mlkn(X, upto=0)
X_eval_hidden_ = mlkn.evaluate(X, layer=0)
print(X_eval_hidden, X_eval_hidden_)

X_eval = mlkn_ensemble(X)
X_eval_ = mlkn_ensemble.evaluate(X)
print(X_eval, X_eval_)

X_eval_hidden = mlkn_ensemble(X, upto=0)
X_eval_hidden_ = mlkn_ensemble.evaluate(X, layer=0)
print(X_eval_hidden, X_eval_hidden_)
