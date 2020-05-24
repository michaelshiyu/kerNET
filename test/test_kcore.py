"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import unittest

import numpy as np

import torch
import torch.nn.functional as F

import kernet_future.utils as utils
from kernet_future.layers.kcore import Phi


os.environ['KMP_DUPLICATE_LIB_OK']='True'  # fix an OMP bug on MacOS


class PhiTest(unittest.TestCase): 

  def setUp(self):
    # allow tests to be run on GPU if possible
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_features = 3
    self.gaussian_phi = Phi(
        kernel='gaussian',
        evaluation='indirect',
        sigma=1.
        )
    self.nn_tanh_phi = Phi(
        in_features=in_features,
        kernel='nn_tanh',
        evaluation='direct',
        )
    self.nn_sigmoid_phi = Phi(
        in_features=in_features,
        kernel='nn_sigmoid',
        evaluation='direct',
        )
    self.nn_relu_phi = Phi(
      in_features=in_features,
      kernel='nn_relu',
      evaluation='direct',
    )
    self.nn_reapen_phi = Phi(
      in_features=in_features,
      kernel='nn_reapen',
      evaluation='direct',
    )
    self.in_features = in_features

  #########
  # test forward
  #########
  def test_gaussian_phi_forward(self):
    # NOTE this also tests gaussian_phi.get_k_mtrx
    input2d = torch.tensor([[.1, .5, .3], [0, .1, -.1]]).to(self.device)
    centers2d = torch.tensor([[0, -.2, .1], [.1, .2, .3]]).to(self.device)
    input3d = torch.tensor([[[.1, .5, .3]], [[0, .1, -.1]]]).to(self.device)
    centers3d = torch.tensor([[[0, -.2, .1]], [[.1, .2, .3]]]).to(self.device)
    input4d = torch.tensor([[[[.1, .5, .3]]], [[[0, .1, -.1]]]]).to(self.device)
    centers4d = torch.tensor([[[[0, -.2, .1]]], [[[.1, .2, .3]]]]).to(self.device)

    for input, centers in zip(
        [input2d, input3d, input4d], 
        [centers2d, centers3d, centers4d]
    ):
      res = self.gaussian_phi(input, centers)
      self.assertTrue(np.allclose(
        res.cpu().numpy(),
        np.array([[0.763379494, 0.955997481], [0.937067463, 0.913931185]])
        ))

  def test_nn_tanh_phi_forward(self):
    input = torch.randn(100, 3).to(self.device)
    res = self.nn_tanh_phi(input)
    target = torch.tanh(input) 
    self.assertTrue(np.allclose(
      res.cpu().numpy(),
      (target / torch.norm(target, dim=1, keepdim=True)).cpu().numpy()
      ))

  def test_nn_sigmoid_phi_forward(self):
    input = torch.randn(100, 3).to(self.device)
    res = self.nn_sigmoid_phi(input)
    target = torch.sigmoid(input)
    self.assertTrue(np.allclose(
      res.cpu().numpy(),
      (target / torch.norm(target, dim=1, keepdim=True)).cpu().numpy()
    ))

  def test_nn_relu_phi_forward(self):
    input = torch.randn(100, 3).to(self.device)
    res = self.nn_relu_phi(input)
    target = F.relu(input)
    self.assertTrue(np.allclose(
      res.cpu().numpy(),
      (target / torch.max(
        utils.AVOID_ZERO_DIV.to(target.device),
        torch.norm(target, dim=1, keepdim=True)
      )).cpu().numpy()
    ))

  def test_nn_reapen_phi_forward(self):
    input = torch.randn(100, 3, 32, 32).to(self.device)
    res = self.nn_reapen_phi(input)
    target = F.avg_pool2d(F.relu(input), 4).view(len(input), -1)
    self.assertTrue(torch.allclose(
      res,
      (target / torch.max(
        utils.AVOID_ZERO_DIV.to(self.device),
        torch.norm(target, dim=1, keepdim=True)
      ))
    ))

  #########
  # test unit norm
  #########
  def test_nn_tanh_phi_norm(self):
    input = torch.randn(100, 3).to(self.device)
    target = torch.tensor([1.] * 100).to(self.device)
    res = self.nn_tanh_phi(input)
    self.assertTrue(np.allclose(
      torch.norm(res, dim=1).cpu().numpy(),
      target.cpu().numpy()
      ))

  def test_nn_sigmoid_phi_norm(self):
    input = torch.randn(100, 3).to(self.device)
    target = torch.tensor([1.] * 100).to(self.device)
    res = self.nn_sigmoid_phi(input)
    self.assertTrue(np.allclose(
      torch.norm(res, dim=1).cpu().numpy(),
      target.cpu().numpy()
    ))

  def test_nn_relu_phi_norm(self):
    input = torch.randn(100, 3).to(self.device)
    res = self.nn_relu_phi(input)
    target = np.where(np.array([any(_ > 0) for _ in res]) > 0, 1., 0.)
    self.assertTrue(np.allclose(
      torch.norm(res, dim=1).cpu().numpy(),
      target,
    ))

  def test_nn_reapen_phi_norm(self):
    input = torch.randn(100, 3, 32, 32).to(self.device)
    res = self.nn_reapen_phi(input)
    self.assertTrue(torch.allclose(
      torch.norm(res, dim=1),
      torch.tensor([1.] * len(input)).to(self.device)
    ))

  #########
  # test kernel matrix
  #########
  def test_nn_tanh_phi_get_k_mtrx(self):
    input1 = torch.tensor([[.1, .5, .3], [0, .1, -.1]]).to(self.device)
    input2 = torch.tensor([[0, -.2, .1], [.1, .2, .3]]).to(self.device)

    res = self.nn_tanh_phi.get_k_mtrx(input1, input2)
    self.assertTrue(np.allclose(
      res.cpu().numpy(),
      # np.array([[-0.062175978, 0.186007269], [-0.029605711, -0.009362541]]) # unnormalized
      np.array([[-0.506393473, 0.915915961], [-0.949929847, -0.181622650]])
      ))

  def test_nn_sigmoid_phi_get_k_mtrx(self):
    input1 = torch.tensor([[.1, .5, .3], [0, .1, -.1]]).to(self.device)
    input2 = torch.tensor([[0, -.2, .1], [.1, .2, .3]]).to(self.device)

    res = self.nn_sigmoid_phi.get_k_mtrx(input1, input2)
    self.assertTrue(np.allclose(
      res.cpu().numpy(),
      # np.array([[0.844269988, 0.947836654], [0.735703822, 0.824013149]]) # unnormalized
      np.array([[0.992787534, 0.998227035], [0.994650158, 0.997751410]])
      ))

  def test_nn_relu_phi_get_k_mtrx(self):
    input1 = torch.tensor([[.1, .5, .3], [0, .1, -.1]]).to(self.device)
    input2 = torch.tensor([[0, -.2, .1], [.1, .2, .3]]).to(self.device)

    res = self.nn_relu_phi.get_k_mtrx(input1, input2)
    self.assertTrue(np.allclose(
      res.cpu().numpy(),
      np.array([[0.50709255283711, 0.9035079029052513], [0, 0.5345224838248488]])
    ))

  def test_nn_reapen_phi_get_k_mtrx(self):
    input1 = torch.randn(10, 3, 32, 32).to(self.device)
    input2 = torch.randn(10, 3, 32, 32).to(self.device)
    res = self.nn_reapen_phi.get_k_mtrx(input1, input2)
    phi1 = F.avg_pool2d(F.relu(input1), 4).view(len(input1), -1)
    phi1_normalized = phi1 / torch.norm(phi1, dim=1, keepdim=True)
    phi2 = F.avg_pool2d(F.relu(input2), 4).view(len(input2), -1)
    phi2_normalized = phi2 / torch.norm(phi2, dim=1, keepdim=True)
    self.assertTrue(torch.allclose(
      res,
      phi1_normalized.mm(phi2_normalized.t())
    ))

  #########
  # test ideal kernel matrix
  #########
  def test_gaussian_phi_get_ideal_k_mtrx(self):
    target1_1 = torch.tensor([1, 0, 2]).to(self.device)
    target1_2 = torch.tensor([1, 1, 0]).to(self.device)
    target2_1 = torch.tensor([[1], [0], [2]]).to(self.device)
    target2_2 = torch.tensor([[1], [1], [0]]).to(self.device)

    for target1, target2 in zip(
        [target1_1, target2_1],
        [target1_2, target2_2]):
      ideal_k_mtrx = self.gaussian_phi.get_ideal_k_mtrx(
          target1,
          target2,
          n_classes=3
          )
      self.assertTrue(np.all(np.equal(
        ideal_k_mtrx.cpu().numpy(),
        np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])
        )))

  def test_nn_tanh_phi_get_ideal_k_mtrx(self):
    target1_1 = torch.tensor([1, 0, 2]).to(self.device)
    target1_2 = torch.tensor([1, 1, 0]).to(self.device)
    target2_1 = torch.tensor([[1], [0], [2]]).to(self.device)
    target2_2 = torch.tensor([[1], [1], [0]]).to(self.device)

    for target1, target2 in zip(
        [target1_1, target2_1],
        [target1_2, target2_2]):
      ideal_k_mtrx = self.nn_tanh_phi.get_ideal_k_mtrx(
          target1,
          target2,
          n_classes=3
          )
      self.assertTrue(np.all(np.equal(
        ideal_k_mtrx.cpu().numpy(),
        np.array([
          [1, 1, -1], 
          [-1, -1, 1], 
          [-1, -1, -1]
          ]))))

  def test_nn_sigmoid_phi_get_ideal_k_mtrx(self):
    target1_1 = torch.tensor([1, 0, 2]).to(self.device)
    target1_2 = torch.tensor([1, 1, 0]).to(self.device)
    target2_1 = torch.tensor([[1], [0], [2]]).to(self.device)
    target2_2 = torch.tensor([[1], [1], [0]]).to(self.device)

    for target1, target2 in zip(
        [target1_1, target2_1],
        [target1_2, target2_2]):
      ideal_k_mtrx = self.nn_sigmoid_phi.get_ideal_k_mtrx(
          target1,
          target2,
          n_classes=3
          )
      self.assertTrue(np.all(np.equal(
        ideal_k_mtrx.cpu().numpy(),
        np.array([
          [1, 1, 0], 
          [0, 0, 1], 
          [0, 0, 0]
          ]))))

  def test_nn_relu_phi_get_ideal_k_mtrx(self):
    target1_1 = torch.tensor([1, 0, 2]).to(self.device)
    target1_2 = torch.tensor([1, 1, 0]).to(self.device)
    target2_1 = torch.tensor([[1], [0], [2]]).to(self.device)
    target2_2 = torch.tensor([[1], [1], [0]]).to(self.device)

    for target1, target2 in zip(
        [target1_1, target2_1],
        [target1_2, target2_2]):
      ideal_k_mtrx = self.nn_relu_phi.get_ideal_k_mtrx(
        target1,
        target2,
        n_classes=3
      )
      self.assertTrue(np.all(np.equal(
        ideal_k_mtrx.cpu().numpy(),
        np.array([
          [1, 1, 0],
          [0, 0, 1],
          [0, 0, 0]
        ]))))

  def test_nn_reapen_phi_get_ideal_k_mtrx(self):
    target1_1 = torch.tensor([1, 0, 2]).to(self.device)
    target1_2 = torch.tensor([1, 1, 0]).to(self.device)
    target2_1 = torch.tensor([[1], [0], [2]]).to(self.device)
    target2_2 = torch.tensor([[1], [1], [0]]).to(self.device)

    for target1, target2 in zip(
        [target1_1, target2_1],
        [target1_2, target2_2]):
      ideal_k_mtrx = self.nn_reapen_phi.get_ideal_k_mtrx(
        target1,
        target2,
        n_classes=3
      )
      self.assertTrue(np.all(np.equal(
        ideal_k_mtrx.cpu().numpy(),
        np.array([
          [1, 1, 0],
          [0, 0, 1],
          [0, 0, 0]
        ]))))
