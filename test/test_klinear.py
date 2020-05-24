"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import unittest

import numpy as np

import torch

import kernet_future.utils as utils
from kernet_future.models import Normalize, Flatten
from kernet_future.layers.loss import SRSLoss
from kernet_future.layers.klinear import kLinear


class kLinearLayerTest(unittest.TestCase):

  def setUp(self):
    # allow tests to be run on GPU if possible
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.input = torch.randn(100, 15).to(self.device)
    self.target = torch.randint(0, 10, (100,)).to(self.device)
    self.klinear = kLinear(out_features=10, kernel='gaussian', evaluation='indirect', centers=self.input).to(self.device)
    self.klinear_committee = utils.to_committee(self.klinear, 30)

    # toy two-layer model
    self.toy_input = torch.tensor([[1., 2], [3, 4]]).to(self.device)
    self.toy_target = torch.tensor([0, 1]).to(self.device)
    self.toy_klinear1 = kLinear(
      out_features=2,
      kernel='gaussian', evaluation='indirect',
      centers=self.toy_input, sigma=3
    ).to(self.device)
    self.toy_klinear1.linear.weight.data = torch.tensor([[.1, .2], [.5, .7]]).to(self.device)
    self.toy_klinear1.linear.bias.data = torch.tensor([0., 0]).to(self.device)

    self.toy_klinear2 = kLinear(
      out_features=2,
      kernel='gaussian', evaluation='indirect',
      centers=self.toy_klinear1(self.toy_input).detach(), sigma=2
    ).to(self.device)  # stop grad flow through layer2 centers to layer1
    self.toy_klinear2.linear.weight.data = torch.tensor([[1.2, .3], [.2, 1.7]]).to(self.device)
    self.toy_klinear2.linear.bias.data = torch.tensor([.1, .2]).to(self.device)

    self.toy_net = torch.nn.Sequential(
      self.toy_klinear1,
      self.toy_klinear2
    )

  def test_two_layer_gaussian_indirect_forward(self, ):
    # kLinear with Gaussian kernel does not have to work with
    # 2d input only
    input2d = self.toy_input
    input3d = input2d.unsqueeze(1)
    input4d = input3d.unsqueeze(1)

    for input in [input2d, input3d, input4d]:
      # change centers accordingly to prevent dimension mismatch
      self.toy_klinear1.centers = input
      self.toy_klinear2.centers = self.toy_klinear1(input).detach() # stop grad flow through layer2 centers to layer1

      self.assertTrue(np.allclose(
        self.toy_net(input).detach().cpu().numpy(),
        np.array([[1.5997587, 2.0986326], [1.5990349, 2.0998392]])
        ))
      self.assertTrue(np.allclose(
        self.toy_klinear1(input).detach().cpu().numpy(),
        np.array([[0.22823608, 0.9488263], [0.26411805, 1.0205902]])
        ))
      
  def test_nn_tanh_kernel_consistency_with_native_pytorch(self, ):
    input = torch.randn(30, 784).to(self.device)

    klinear1 = kLinear(
        in_features=784, out_features=10,
        kernel='nn_tanh', evaluation='direct',
        ).to(self.device)

    linear = torch.nn.Linear(784, 10).to(self.device)
    linear.weight.data = klinear1.linear.weight.data
    linear.bias.data = klinear1.linear.bias.data

    torch_model = torch.nn.Sequential(
        torch.nn.Tanh(),
        Normalize(),
        linear
        )

    self.assertTrue(np.allclose(
      klinear1(input).detach().cpu().numpy(),
      torch_model(input).detach().cpu().numpy()
      ))

  def test_nn_sigmoid_kernel_consistency_with_native_pytorch(self, ):
    input = torch.randn(30, 784).to(self.device)

    klinear1 = kLinear(
        in_features=784, out_features=10,
        kernel='nn_sigmoid', evaluation='direct',
        ).to(self.device)

    linear = torch.nn.Linear(784, 10).to(self.device)
    linear.weight.data = klinear1.linear.weight.data
    linear.bias.data = klinear1.linear.bias.data

    torch_model = torch.nn.Sequential(
        torch.nn.Sigmoid(),
        Normalize(),
        linear
        )

    self.assertTrue(np.allclose(
      klinear1(input).detach().cpu().numpy(),
      torch_model(input).detach().cpu().numpy()
      ))

  def test_nn_relu_kernel_consistency_with_native_pytorch(self, ):
    input = torch.randn(30, 784).to(self.device)

    klinear1 = kLinear(
      in_features=784, out_features=10,
      kernel='nn_relu', evaluation='direct',
    ).to(self.device)

    linear = torch.nn.Linear(784, 10).to(self.device)
    linear.weight.data = klinear1.linear.weight.data
    linear.bias.data = klinear1.linear.bias.data

    torch_model = torch.nn.Sequential(
      torch.nn.ReLU(),
      Normalize(),
      linear
    )

    self.assertTrue(np.allclose(
      klinear1(input).detach().cpu().numpy(),
      torch_model(input).detach().cpu().numpy()
    ))

  def test_nn_reapen_kernel_consistency_with_native_pytorch(self, ):
    input = torch.randn(30, 3, 32, 32).to(self.device)

    klinear1 = kLinear(
      in_features=192, out_features=10,
      kernel='nn_reapen', evaluation='direct',
    ).to(self.device)

    linear = torch.nn.Linear(192, 10).to(self.device)
    linear.weight.data = klinear1.linear.weight.data
    linear.bias.data = klinear1.linear.bias.data

    torch_model = torch.nn.Sequential(
      torch.nn.ReLU(),
      torch.nn.AvgPool2d(4),
      Flatten(),
      Normalize(),
      linear
    )

    self.assertTrue(torch.allclose(
      klinear1(input),
      torch_model(input),
    ))

  def test_single_model_equals_committee(self):
    single_model_output = self.klinear(self.input).detach().cpu().numpy()
    committee_output = self.klinear_committee(self.input).detach().cpu().numpy()
    self.assertTrue(np.allclose(
      single_model_output,
      committee_output,
      atol=1e-5
    ))

  def test_single_model_equals_committee_in_backward_pass(self):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(self.klinear.parameters())
    committee_optimizer = torch.optim.Adam(self.klinear_committee.parameters())

    for i in range(10):
      optimizer.zero_grad()
      committee_optimizer.zero_grad()

      loss = loss_fn(
        self.klinear(self.input),
        self.target
      )
      loss.backward()
      optimizer.step()

      loss = loss_fn(
        self.klinear_committee(self.input),
        self.target
      )
      loss.backward()
      committee_optimizer.step()

    single_model_output = self.klinear(self.input).detach().cpu().numpy()
    committee_output = self.klinear_committee(self.input).detach().cpu().numpy()
    self.assertTrue(np.allclose(
      single_model_output,
      committee_output,
      atol=1e-5
    ))

  def test_grad_computation_is_correct(self):
    map_input = lambda x: self.toy_klinear2.phi.get_k_mtrx(x, x)
    map_target = lambda x: self.toy_klinear2.phi.get_ideal_k_mtrx(x, x, n_classes=2)
    loss_fn = torch.nn.CosineSimilarity(dim=0)
    loss_fn1 = SRSLoss(
      map_input=map_input,
      map_target=map_target,
      loss=loss_fn
    )
    loss_fn2 = torch.nn.CrossEntropyLoss()

    dummy_optimizer = torch.optim.Adam(self.toy_net.parameters())
    dummy_optimizer.zero_grad() # make sure grad is zeroed

    loss = loss_fn1(
      self.toy_klinear1(self.toy_input),
      self.toy_target
    )
    loss.backward()

    # freeze layer1
    for p in self.toy_klinear1.parameters():
      p.requires_grad_(False)

    loss = loss_fn2(
      self.toy_net(self.toy_input),
      self.toy_target
    )
    loss.backward()

    grad_w1 = self.toy_klinear1.linear.weight.grad.detach().cpu().numpy()
    grad_b1 = self.toy_klinear1.linear.bias.grad.detach().cpu().numpy()
    grad_w2 = self.toy_klinear2.linear.weight.grad.detach().cpu().numpy()
    grad_b2 = self.toy_klinear2.linear.bias.grad.detach().cpu().numpy()

    self.assertTrue(np.allclose(
      grad_w1,
      np.array([[-0.00113756, 0.00113756], [-0.00227511, 0.00227511]])
    ))
    self.assertTrue(np.allclose(
      grad_b1,
      np.array([0., 0.])
    ))
    self.assertTrue(np.allclose(
      grad_w2,
      np.array([[-0.12257326, -0.12217124], [0.12257326, 0.12217124]])
    ))
    self.assertTrue(np.allclose(
      grad_b2,
      np.array([-0.12242149, 0.12242149])
    ))
