#!/usr/bin/env python

import unittest
import numpy as np
import torch

import kernet.backend as K
from kernet.models.ffc import _baseFFC, FFC, greedyFFC
from kernet.layers.kn import knFC, knFCEnsemble

torch.manual_seed(1234)
np.random.seed(1234)

# forward test for _baseFFC done, bp fit does not need testing since it is
# simply a wrapper around native pytorch bp

# greedyFFC with knFC and knFCEnsemble
# tested forward and initial grad calculations on the toy example,
# does not test updated weights since updating the first layer would change
# grad of the second

class KNTestCase(unittest.TestCase):
    def setUp(self):

        #########
        # toy data

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.X = torch.tensor([[1, 2], [3, 4]], dtype=torch.float, device=device)
        self.Y = torch.tensor([0, 1], dtype=torch.int64, device=device)

        #########
        # base kn
        self.kn = greedyFFC()
        self.kn.add_layer(knFC(
            X=self.X,
            n_out=2,
            kernel='gaussian',
            sigma=3,
            bias=True
        ))
        self.kn.add_layer(knFC(
            X=self.X,
            n_out=2,
            kernel='gaussian',
            sigma=2,
            bias=True
        ))
        
        # manually set some weights
        self.kn.layer0.weight.data = torch.Tensor([[.1, .2], [.5, .7]])
        self.kn.layer0.bias.data = torch.Tensor([0., 0.])
        self.kn.layer1.weight.data = torch.Tensor([[1.2, .3], [.2, 1.7]])
        self.kn.layer1.bias.data = torch.Tensor([0.1, 0.2])

        self.kn.add_critic(self.kn.layer1.phi)
        self.kn.add_loss(torch.nn.CosineSimilarity())
        self.kn.add_metric(torch.nn.CosineSimilarity())
        self.kn.add_loss(torch.nn.CrossEntropyLoss())
        self.kn.add_metric(torch.nn.CrossEntropyLoss())

        #########
        # ensemble

        self.kn_ensemble = greedyFFC()
        self.kn_ensemble.add_layer(K.to_ensemble(self.kn.layer0, batch_size=1))
        self.kn_ensemble.add_layer(K.to_ensemble(self.kn.layer1, batch_size=1))
        self.kn_ensemble.add_critic(self.kn.layer1.phi)
        self.kn_ensemble.add_loss(torch.nn.CosineSimilarity())
        self.kn_ensemble.add_metric(torch.nn.CosineSimilarity())
        self.kn_ensemble.add_loss(torch.nn.CrossEntropyLoss())
        self.kn_ensemble.add_metric(torch.nn.CrossEntropyLoss())

        self.kn.to(device)
        self.kn_ensemble.to(device)


    def test_forward_and_evaluate(self):
        # test forward
        X_eval = self.kn(self.X, update_X=True)
        X_eval_hidden = self.kn(self.X, update_X=True, upto=0)
        self.assertTrue(np.allclose(
            X_eval.detach().to('cpu').numpy(),
            np.array([[1.5997587, 2.0986326], [1.5990349, 2.0998392]])
            ))
        self.assertTrue(np.allclose(
            X_eval_hidden.detach().to('cpu').numpy(),
            np.array([[0.22823608, 0.9488263], [0.26411805, 1.0205902]])
            ))
        # test forward equals evaluate
        X_eval_ = self.kn.evaluate(self.X)
        X_eval_hidden_ = self.kn.evaluate(self.X, layer=0)
        self.assertTrue(np.array_equal(
            X_eval.detach().to('cpu').numpy(),
            X_eval_.detach().to('cpu').numpy()
            ))
        self.assertTrue(np.array_equal(
            X_eval_hidden.detach().to('cpu').numpy(),
            X_eval_hidden_.detach().to('cpu').numpy()
            ))

    def test_ensemble_forward_and_evaluate(self):
        # test forward for ensemble
        X_eval = self.kn_ensemble(self.X, update_X=True)
        X_eval_hidden = self.kn_ensemble(self.X, update_X=True, upto=0)
        # print(X_eval.detach().to('cpu').numpy())
        self.assertTrue(np.allclose(
            X_eval.detach().to('cpu').numpy(),
            np.array([[1.5997587, 2.0986326], [1.5990349, 2.0998392]])
            ))
        self.assertTrue(np.allclose(
            X_eval_hidden.detach().to('cpu').numpy(),
            np.array([[0.22823608, 0.9488263], [0.26411805, 1.0205902]])
            ))
        # test forward equals evaluate for ensemble
        X_eval_ = self.kn_ensemble.evaluate(self.X)
        X_eval_hidden_ = self.kn_ensemble.evaluate(self.X, layer=0)
        self.assertTrue(np.array_equal(
            X_eval.detach().to('cpu').numpy(),
            X_eval_.detach().to('cpu').numpy()
            ))
        self.assertTrue(np.array_equal(
            X_eval_hidden.detach().to('cpu').numpy(),
            X_eval_hidden_.detach().to('cpu').numpy()
            ))

    def test_grad(self):
        self.kn.add_optimizer(torch.optim.SGD(self.kn.parameters(), lr=0))
        self.kn.add_optimizer(torch.optim.SGD(self.kn.parameters(), lr=0))

        self.kn.fit(
            n_epoch=(1, 1),
            X=self.X,
            Y=self.Y,
            n_class=2,
            keep_grad=True,
            verbose=False
            )

        self.assertTrue(np.allclose(
            self.kn.layer0.weight.grad.detach().to('cpu').numpy(),
            np.array([[0.00113756, -0.00113756], [0.00227511, -0.00227511]])
            ))
        self.assertTrue(np.allclose(
            self.kn.layer0.bias.grad.detach().to('cpu').numpy(),
            np.array([0., 0.])
            ))
        self.assertTrue(np.allclose(
            self.kn.layer1.weight.grad.detach().to('cpu').numpy(),
            np.array([[-0.12257326, -0.12217124], [0.12257326, 0.12217124]])
            ))
        self.assertTrue(np.allclose(
            self.kn.layer1.bias.grad.detach().to('cpu').numpy(),
            np.array([-0.12242149, 0.12242149])
            ))

    def test_ensemble_grad(self):
        self.kn_ensemble.add_optimizer(torch.optim.SGD(self.kn_ensemble.parameters(), lr=0))
        self.kn_ensemble.add_optimizer(torch.optim.SGD(self.kn_ensemble.parameters(), lr=0))

        self.kn_ensemble.fit(
            n_epoch=(1, 1),
            X=self.X,
            Y=self.Y,
            n_class=2,
            keep_grad=True,
            verbose=False
            )

        w0_grad, b0_grad, w1_grad, b1_grad = [], [], [], []
        for w in self.kn_ensemble.layer0.weight:
            w0_grad.append(w.grad.detach().to('cpu').numpy())
        for b in self.kn_ensemble.layer0.bias:
            if b is not None:
                b0_grad.append(b.grad.detach().to('cpu').numpy())
        for w in self.kn_ensemble.layer1.weight:
            w1_grad.append(w.grad.detach().to('cpu').numpy())
        for b in self.kn_ensemble.layer1.bias:
            if b is not None:
                b1_grad.append(b.grad.detach().to('cpu').numpy())
        # print(w0_grad)
        self.assertTrue(np.allclose(
            w0_grad,
            [np.array([[0.00113756], [0.00227511]]),
            np.array([[-0.00113756], [-0.00227511]])]
            ))
        self.assertTrue(np.allclose(
            w1_grad,
            [np.array([[-0.12257326], [0.12257326]]),
            np.array([[-0.12217125], [0.12217125]])]
            ))
        self.assertTrue(np.allclose(
            b0_grad,
            [np.array([0., 0.])]
            ))
        self.assertTrue(np.allclose(
            b1_grad,
            [np.array([-0.1224215,  0.1224215])]
            ))

    def test_forward_and_evaluate_after_training(self):
        self.kn.add_optimizer(torch.optim.SGD(self.kn.parameters(), lr=1))
        self.kn.add_optimizer(torch.optim.SGD(self.kn.parameters(), lr=1))

        self.kn_ensemble.add_optimizer(torch.optim.SGD(self.kn_ensemble.parameters(), lr=1))
        self.kn_ensemble.add_optimizer(torch.optim.SGD(self.kn_ensemble.parameters(), lr=1))

        self.kn.fit(
            n_epoch=(10, 10),
            X=self.X,
            Y=self.Y,
            n_class=2,
            keep_grad=True,
            verbose=False
            )

        self.kn_ensemble.fit(
            n_epoch=(10, 10),
            X=self.X,
            Y=self.Y,
            n_class=2,
            keep_grad=True,
            verbose=False
            )

        # test forward equals evaluate
        X_eval = self.kn(self.X, update_X=True)
        X_eval_ = self.kn.evaluate(self.X)
        self.assertTrue(np.array_equal(
            X_eval.detach().to('cpu').numpy(),
            X_eval_.detach().to('cpu').numpy()
            ))

        X_eval_hidden = self.kn(self.X, upto=0, update_X=True)
        X_eval_hidden_ = self.kn.evaluate(self.X, layer=0)
        self.assertTrue(np.array_equal(
            X_eval_hidden.detach().to('cpu').numpy(),
            X_eval_hidden_.detach().to('cpu').numpy()
            ))

        # test forward equals evaluate for ensemble
        X_eval_ensemble = self.kn_ensemble(self.X, update_X=True)
        X_eval_ensemble_ = self.kn_ensemble.evaluate(self.X)
        self.assertTrue(np.array_equal(
            X_eval_ensemble.detach().to('cpu').numpy(),
            X_eval_ensemble_.detach().to('cpu').numpy()
            ))

        X_eval_hidden_ensemble = self.kn_ensemble(self.X, upto=0, update_X=True)
        X_eval_hidden_ensemble_ = self.kn_ensemble.evaluate(self.X, layer=0)
        self.assertTrue(np.array_equal(
            X_eval_hidden_ensemble.detach().to('cpu').numpy(),
            X_eval_hidden_ensemble_.detach().to('cpu').numpy()
            ))

        # test ensemble equals ordinary
        self.assertTrue(np.allclose(
            X_eval.detach().to('cpu').numpy(),
            X_eval_ensemble.detach().to('cpu').numpy()
            ))
        self.assertTrue(np.allclose(
            X_eval_hidden.detach().to('cpu').numpy(),
            X_eval_hidden_ensemble.detach().to('cpu').numpy()
            ))

if __name__=='__main__':
    unittest.main()
