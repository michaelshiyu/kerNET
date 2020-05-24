"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import unittest

from kernet_future.layers.loss import *
from kernet_future.layers.kcore import Phi


class SRSLossTest(unittest.TestCase):

  def setUp(self):
    # allow tests to be run on GPU if possible
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.input_4D = torch.tensor([[1, 3.], [.5, -2], [9, .1]]).view(3, 1, 2, 1).to(self.device)
    self.input = torch.tensor([[1, 3.], [.5, -2], [9, .1]]).to(self.device)
    self.target1 = torch.tensor([0, 1, 0]).to(self.device)
    self.target2 = torch.tensor([[0], [1], [0]]).to(self.device)
    self.n_classes = 2

    self.phi = Phi(
      kernel='gaussian',
      evaluation='indirect',
      sigma=10.
    )
    self.another_phi = Phi(
      kernel='nn_tanh',
      evaluation='direct'
    )

  #########
  # test instantiations with arbitrary map_input, map_target, and loss
  #########

  def test_srsloss_instance1_2dinput(self, ):
    map_input = lambda x: torch.sum((x - x.unsqueeze(1)) ** 2, dim=(-1)).unsqueeze(-1)
    map_target = lambda x: x * x.unsqueeze(1)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    srsloss_fn = SRSLoss(map_input=map_input, map_target=map_target, loss=loss_fn)

    input = torch.tensor([[1., 2], [3, 1], [5, 6]]).to(self.device)
    target = torch.tensor([[1.], [-1], [1]]).to(self.device)

    res = srsloss_fn(input, target)
    self.assertAlmostEqual(res.cpu().numpy(), 421.8889, places=4)

  def test_srsloss_instance1_3dinput(self, ):
    map_input = lambda x: torch.sum((x - x.unsqueeze(1)) ** 2, dim=(-1, -2)).unsqueeze(-1)    
    map_target = lambda x: x * x.unsqueeze(1)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    srsloss_fn = SRSLoss(map_input=map_input, map_target=map_target, loss=loss_fn)

    input = torch.tensor([[[1., 2], [1, 3]], [[3, 1], [6, 4]], [[5, 6], [0, 0]]]).to(self.device)
    target = torch.tensor([[1.], [-1], [-1]]).to(self.device)

    res = srsloss_fn(input, target)
    self.assertAlmostEqual(res.cpu().numpy(), 2061., places=4)

  def test_srsloss_instance1_4dinput(self, ):
    map_input = lambda x: torch.sum((x - x.unsqueeze(1)) ** 2, dim=(-1, -2, -3)).unsqueeze(-1)
    map_target = lambda x: x * x.unsqueeze(1)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    srsloss_fn = SRSLoss(map_input=map_input, map_target=map_target, loss=loss_fn)

    input = torch.tensor([[[[1., 2], [1, 3]]], [[[3, 1], [6, 4]]], [[[5, 6], [0, 0]]]]).to(self.device)
    target = torch.tensor([[1.], [-1], [-1]]).to(self.device)

    res = srsloss_fn(input, target)
    self.assertAlmostEqual(res.cpu().numpy(), 2061., places=4)

  def test_srsloss_instance1_2drandom_input(self, ):
    map_input = lambda x: torch.sum((x - x.unsqueeze(1)) ** 2, dim=(-1)).unsqueeze(-1)
    map_target = lambda x: x * x.unsqueeze(1)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    srsloss_fn = SRSLoss(map_input=map_input, map_target=map_target, loss=loss_fn)

    input = torch.randn(100, 28).to(self.device)
    target = torch.randn(100, 1).to(self.device)

    res = srsloss_fn(input, target)
    
    new_input = map_input(input)
    new_target = map_target(target)
    res1 = loss_fn(new_input.view(-1, 1), new_target.view(-1, 1))

    self.assertAlmostEqual(res.cpu().numpy(), res1.cpu().numpy(), places=4)

  def test_srsloss_instance1_3drandom_input(self, ):
    map_input = lambda x: torch.sum((x - x.unsqueeze(1)) ** 2, dim=(-1, -2)).unsqueeze(-1)
    map_target = lambda x: x * x.unsqueeze(1)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    srsloss_fn = SRSLoss(map_input=map_input, map_target=map_target, loss=loss_fn)

    input = torch.randn(100, 28, 28).to(self.device)
    target = torch.randn(100, 1).to(self.device)

    res = srsloss_fn(input, target)
    
    new_input = map_input(input)
    new_target = map_target(target)
    res1 = loss_fn(new_input.view(-1, 1), new_target.view(-1, 1))

    self.assertAlmostEqual(res.cpu().numpy(), res1.cpu().numpy(), places=4)

  def test_srsloss_instance1_4drandom_input(self, ):
    map_input = lambda x: torch.sum((x - x.unsqueeze(1)) ** 2, dim=(-1, -2, -3)).unsqueeze(-1)
    map_target = lambda x: x * x.unsqueeze(1)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    srsloss_fn = SRSLoss(map_input=map_input, map_target=map_target, loss=loss_fn)

    input = torch.randn(100, 3, 28, 28).to(self.device)
    target = torch.randn(100, 1).to(self.device)

    res = srsloss_fn(input, target)
    
    new_input = map_input(input)
    new_target = map_target(target)
    res1 = loss_fn(new_input.view(-1, 1), new_target.view(-1, 1))

    self.assertAlmostEqual(res.cpu().numpy(), res1.cpu().numpy(), places=4)

  #########
  # test reference instantiations
  #########

  def test_srs_raw(self, ):
    srsloss_fn = srs_raw(self.phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), -0.2889, places=4)

  def test_srs_raw_neo(self, ):
    srsloss_fn = srs_raw_neo(self.phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), -0.7815, places=4)

  def test_srs_nmse(self, ):
    srsloss_fn = srs_nmse(self.phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), -0.2964, places=4)

  def test_srs_nmse_neo(self, ):
    srsloss_fn = srs_nmse_neo(self.phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), -0.6207, places=4)

  def test_srs_alignment(self, ):
    srsloss_fn = srs_alignment(self.phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), 0.7733, places=4)

  def test_srs_alignment_neo(self, ):
    self.assertRaises(ValueError, srs_alignment_neo, phi=self.phi, n_classes=self.n_classes)

  def test_srs_upper_tri_alignment(self, ):
    srsloss_fn = srs_upper_tri_alignment(self.phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), 0.5299, places=4)

  def test_srs_upper_tri_alignment_neo(self, ):
    self.assertRaises(ValueError, srs_upper_tri_alignment_neo, phi=self.phi, n_classes=self.n_classes)

  def test_srs_contrastive(self, ):
    srsloss_fn = srs_contrastive(self.phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), 0.3136, places=4)

  def test_srs_contrastive_neo(self, ):
    srsloss_fn = srs_contrastive_neo(self.phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), -2.1957, places=4)

  def test_srs_log_contrastive(self, ):
    srsloss_fn = srs_log_contrastive(self.phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), -1.1597, places=4)

  def test_srs_log_contrastive_neo(self, ):
    srsloss_fn = srs_log_contrastive_neo(self.phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), -0.7865, places=4)

  #########
  # test reference instantiations using another phi
  #########

  def test_srs_raw_using_another_phi(self, ):
    srsloss_fn = srs_raw(self.another_phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), 0.2654, places=4)

  def test_srs_raw_neo_using_another_phi(self, ):
    srsloss_fn = srs_raw_neo(self.another_phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), 0.0563, places=4)

  def test_srs_nmse_using_another_phi(self, ):
    srsloss_fn = srs_nmse(self.another_phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), -0.4881, places=4)

  def test_srs_nmse_neo_using_another_phi(self, ):
    srsloss_fn = srs_nmse_neo(self.another_phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), -1.0481, places=4)

  def test_srs_alignment_using_another_phi(self, ):
    srsloss_fn = srs_alignment(self.another_phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), 0.7155, places=4)

  def test_srs_alignment_neo_using_another_phi(self, ):
    srsloss_fn = srs_alignment_neo(self.another_phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), 0.1405, places=4)

  def test_srs_upper_tri_alignment_using_another_phi(self, ):
    srsloss_fn = srs_upper_tri_alignment(self.another_phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), 0.5176, places=4)

  def test_srs_upper_tri_alignment_neo_using_another_phi(self, ):
    srsloss_fn = srs_upper_tri_alignment_neo(self.another_phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), 0.1405, places=4)

  def test_srs_contrastive_using_another_phi(self, ):
    srsloss_fn = srs_contrastive(self.another_phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), 0.4925, places=4)

  def test_srs_contrastive_neo_using_another_phi(self, ):
    srsloss_fn = srs_contrastive_neo(self.another_phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), -1.0207, places=4)

  def test_srs_log_contrastive_using_another_phi(self, ):
    srsloss_fn = srs_log_contrastive(self.another_phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), -0.7083, places=4)

  def test_srs_log_contrastive_neo_using_another_phi(self, ):
    srsloss_fn = srs_log_contrastive_neo(self.another_phi, n_classes=self.n_classes)
    for target in [self.target1, self.target2]:
      res = srsloss_fn(self.input, target)
      self.assertAlmostEqual(res.cpu().numpy(), -0.0205, places=4)
