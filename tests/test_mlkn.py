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
# TODO: maybe test updated weights?

class BaseMLKNTestCase(unittest.TestCase):
    def setUp(self):

        #########
        # toy data
        dtypeX, dtypeY = torch.FloatTensor, torch.LongTensor
        # if torch.cuda.is_available():
        #     dtypeX, dtypeY = torch.cuda.FloatTensor, torch.cuda.LongTensor
        # TODO: if set this to true, will have to take care of converting many
        # results in tests to cpu first before converting to numpy for comparison
        # maybe should test on GPU as well

        self.X = Variable(
            torch.FloatTensor([[1, 2], [3, 4]]).type(dtypeX),
            requires_grad=False
            )
        self.Y = Variable(
            torch.FloatTensor([[0], [1]]).type(dtypeY),
            requires_grad=False
            )

        #########
        # base mlkn
        self.mlkn = MLKNClassifier()
        self.mlkn.add_layer(kerLinear(
            X=self.X,
            out_dim=2,
            sigma=3,
            bias=True
        ))
        self.mlkn.add_layer(kerLinear(
            X=self.X,
            out_dim=2,
            sigma=2,
            bias=True
        ))
        # manually set some weights
        self.mlkn.layer0.weight.data = torch.FloatTensor([[.1, .2], [.5, .7]])
        self.mlkn.layer0.bias.data = torch.FloatTensor([0, 0])
        self.mlkn.layer1.weight.data = torch.FloatTensor([[1.2, .3], [.2, 1.7]])
        self.mlkn.layer1.bias.data = torch.FloatTensor([0.1, 0.2])

        self.mlkn.add_loss(torch.nn.CrossEntropyLoss())

        #########
        # ensemble

        self.mlkn_ensemble = MLKNClassifier()
        self.mlkn_ensemble.add_layer(K.to_ensemble(self.mlkn.layer0, batch_size=1))
        self.mlkn_ensemble.add_layer(K.to_ensemble(self.mlkn.layer1, batch_size=1))

        self.mlkn_ensemble.add_loss(torch.nn.CrossEntropyLoss())


    def test_forward_and_evaluate(self):

        X_eval = self.mlkn(self.X, update_X=True)
        X_eval_hidden = self.mlkn(self.X, update_X=True, upto=0)
        self.assertTrue(np.allclose(
            X_eval.data.numpy(),
            np.array([[1.5997587, 2.0986326], [1.5990349, 2.0998392]])
            ))
        self.assertTrue(np.allclose(
            X_eval_hidden.data.numpy(),
            np.array([[0.22823608, 0.9488263], [0.26411805, 1.0205902]])
            ))

        X_eval = self.mlkn.evaluate(self.X)
        X_eval_hidden = self.mlkn.evaluate(self.X, layer=0)
        self.assertTrue(np.allclose(
            X_eval.data.numpy(),
            np.array([[1.5997587, 2.0986326], [1.5990349, 2.0998392]])
            ))
        self.assertTrue(np.allclose(
            X_eval_hidden.data.numpy(),
            np.array([[0.22823608, 0.9488263], [0.26411805, 1.0205902]])
            ))

    def test_ensemble_forward_and_evaluate(self):

        X_eval = self.mlkn_ensemble(self.X, update_X=True)
        X_eval_hidden = self.mlkn_ensemble(self.X, update_X=True, upto=0)
        self.assertTrue(np.allclose(
            X_eval.data.numpy(),
            np.array([[1.5997587, 2.0986326], [1.5990349, 2.0998392]])
            ))
        self.assertTrue(np.allclose(
            X_eval_hidden.data.numpy(),
            np.array([[0.22823608, 0.9488263], [0.26411805, 1.0205902]])
            ))

        X_eval = self.mlkn_ensemble.evaluate(self.X)
        X_eval_hidden = self.mlkn_ensemble.evaluate(self.X, layer=0)
        self.assertTrue(np.allclose(
            X_eval.data.numpy(),
            np.array([[1.5997587, 2.0986326], [1.5990349, 2.0998392]])
            ))
        self.assertTrue(np.allclose(
            X_eval_hidden.data.numpy(),
            np.array([[0.22823608, 0.9488263], [0.26411805, 1.0205902]])
            ))

    def test_grad(self):
        self.mlkn.add_optimizer(torch.optim.SGD(self.mlkn.parameters(), lr=0))
        self.mlkn.add_optimizer(torch.optim.SGD(self.mlkn.parameters(), lr=0))

        self.mlkn.fit(
            n_epoch=(1, 1),
            X=self.X,
            Y=self.Y,
            n_class=2,
            keep_grad=True,
            verbose=False
            )

        # print(self.mlkn.layer0.weight.grad.data)
        # print(self.mlkn.layer0.bias.grad.data)
        # print(self.mlkn.layer1.weight.grad.data)
        # print(self.mlkn.layer1.bias.grad.data)

    def test_ensemble_grad(self):
        self.mlkn_ensemble.add_optimizer(torch.optim.SGD(self.mlkn_ensemble.parameters(), lr=0))
        self.mlkn_ensemble.add_optimizer(torch.optim.SGD(self.mlkn_ensemble.parameters(), lr=0))

        self.mlkn_ensemble.fit(
            n_epoch=(1, 1),
            X=self.X,
            Y=self.Y,
            n_class=2,
            keep_grad=True,
            verbose=False
            )
        """
        for w in self.mlkn_ensemble.layer0.weight:
            print(w.grad.data)
        for b in self.mlkn_ensemble.layer0.bias:
            if b is not None:
                print(b.grad.data)
        for w in self.mlkn_ensemble.layer1.weight:
            print(w.grad.data)
        for b in self.mlkn_ensemble.layer1.bias:
            if b is not None:
                print(b.grad.data)
        """
    def test_ensemble_equal_to_ordinary_in_training(self):
        self.mlkn.add_optimizer(torch.optim.SGD(self.mlkn.parameters(), lr=1))
        self.mlkn.add_optimizer(torch.optim.SGD(self.mlkn.parameters(), lr=1))

        self.mlkn_ensemble.add_optimizer(torch.optim.SGD(self.mlkn_ensemble.parameters(), lr=1))
        self.mlkn_ensemble.add_optimizer(torch.optim.SGD(self.mlkn_ensemble.parameters(), lr=1))

        self.mlkn.fit(
            n_epoch=(10, 10),
            X=self.X,
            Y=self.Y,
            n_class=2,
            keep_grad=True,
            verbose=False
            )

        self.mlkn_ensemble.fit(
            n_epoch=(10, 10),
            X=self.X,
            Y=self.Y,
            n_class=2,
            keep_grad=True,
            verbose=False
            )

        X_eval = self.mlkn(self.X)
        X_eval_ = self.mlkn.evaluate(self.X)
        # print(X_eval, X_eval_)

        X_eval_hidden = self.mlkn(self.X, upto=0)
        X_eval_hidden_ = self.mlkn.evaluate(self.X, layer=0)
        # print(X_eval_hidden, X_eval_hidden_)

        X_eval = self.mlkn_ensemble(self.X)
        X_eval_ = self.mlkn_ensemble.evaluate(self.X)
        # print(X_eval, X_eval_)

        X_eval_hidden = self.mlkn_ensemble(self.X, upto=0)
        X_eval_hidden_ = self.mlkn_ensemble.evaluate(self.X, layer=0)
        # print(X_eval_hidden, X_eval_hidden_)






if __name__=='__main__':
    unittest.main()
