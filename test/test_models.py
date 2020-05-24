"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import unittest

from easydict import EasyDict as edict

import torch

from kernet.models import Normalize
from kernet.models.resnet18 import ResNet18
from kernet.models.resnet34 import ResNet34
from kernet.models.resnet50 import ResNet50
from kernet.models.resnet101 import ResNet101
from kernet.models.resnet152 import ResNet152
from kernet.models.resnet18n import ResNet18N
from kernet.models.resnet34n import ResNet34N
from kernet.models.resnet50n import ResNet50N
from kernet_future.models.resnet101n import ResNet101N
from kernet_future.models.resnet152n import ResNet152N
from kernet_future.models.kresnet18 import kResNet18
from kernet_future.models.kresnet34 import kResNet34
from kernet_future.models.kresnet50 import kResNet50
from kernet_future.models.kresnet101 import kResNet101
from kernet_future.models.kresnet152 import kResNet152


class ModelsTest(unittest.TestCase):
  def setUp(self):
    self.opt = edict()  # an object used to simulate the actual opt that contains user-specified params
    self.opt.n_classes = 10
    self.opt.activation = 'reapen'
    self.opt.sigma = 1
    self.opt.memory_efficient = False

    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    self.x = torch.randn(10, 3, 32, 32).to(self.device)
    self.y = torch.randint(0, self.opt.n_classes, (10,)).to(self.device)

    self.resnet18 = ResNet18(self.opt).to(self.device)
    self.resnet34 = ResNet34(self.opt).to(self.device)
    self.resnet50 = ResNet50(self.opt).to(self.device)
    self.resnet101 = ResNet101(self.opt).to(self.device)
    self.resnet152 = ResNet152(self.opt).to(self.device)
    self.resnet_list = [
      self.resnet18,
      self.resnet34,
      self.resnet50,
      self.resnet101,
      self.resnet152
    ]

    self.resnet18n = ResNet18N(self.opt).to(self.device)
    self.resnet34n = ResNet34N(self.opt).to(self.device)
    self.resnet50n = ResNet50N(self.opt).to(self.device)
    self.resnet101n = ResNet101N(self.opt).to(self.device)
    self.resnet152n = ResNet152N(self.opt).to(self.device)
    self.resnetn_list = [
      self.resnet18n,
      self.resnet34n,
      self.resnet50n,
      self.resnet101n,
      self.resnet152n
    ]

    self.kresnet18 = kResNet18(self.opt).to(self.device)
    self.kresnet34 = kResNet34(self.opt).to(self.device)
    self.kresnet50 = kResNet50(self.opt).to(self.device)
    self.kresnet101 = kResNet101(self.opt).to(self.device)
    self.kresnet152 = kResNet152(self.opt).to(self.device)
    self.kresnet_list = [
      self.kresnet18,
      self.kresnet34,
      self.kresnet50,
      self.kresnet101,
      self.kresnet152
    ]

  def test_resnetn_normalizes_penultimate_activations_to_unit_vectors(self):
    for net in self.resnetn_list:
      head = torch.nn.Sequential(
        net.layer1,
        net.layer2,
        net.layer3,
        net.layer4,
        net.layer5
      )
      activations = head(self.x)
      norm = torch.norm(activations, dim=1)
      self.assertTrue(torch.allclose(norm, torch.tensor([1.] * len(self.x)).to(self.device)))

  def test_normalized_resnet_agrees_with_resnetn(self):
    for resnet, resnetn in zip(self.resnet_list, self.resnetn_list):
      # insert a normalization layer in between layer5 and fc of resnet
      head = torch.nn.Sequential(
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
        resnet.layer5
      )
      normalize = Normalize()

      def new_forward(input):
        return resnet.fc(normalize(head(input)))
      resnet.forward = new_forward

      for p1, p2 in zip(resnet.parameters(), resnetn.parameters()):
        p1.data = p2.data  # make the two models share weights
      rn_out, r_out = resnetn(self.x), resnet(self.x)
      self.assertTrue(torch.all(rn_out == r_out))

  def test_kresnet_with_nn_reapen_kernel_agrees_with_resnetn(self):
    for kresnet, resnetn in zip(self.kresnet_list, self.resnetn_list):
      for p1, p2 in zip(kresnet.parameters(), resnetn.parameters()):
        p1.data = p2.data  # make the two models share weights
      r_out, kr_out = resnetn(self.x), kresnet(self.x)
      self.assertTrue(torch.all(r_out == kr_out))

  def test_kresnet_with_nn_reapen_kernel_agrees_with_resnetn_after_training(self):
    for kresnet, resnetn in zip(self.kresnet_list, self.resnetn_list):
      for p1, p2 in zip(kresnet.parameters(), resnetn.parameters()):
        p1.data = p2.data  # make the two models share weights

      # train
      loss_fn = torch.nn.CrossEntropyLoss()
      r_optim = torch.optim.Adam(lr=1e-3, params=resnetn.parameters())
      for i in range(100):
        r_optim.zero_grad()
        loss = loss_fn(resnetn(self.x), self.y)
        r_optim.step()
      kr_optim = torch.optim.Adam(lr=1e-3, params=kresnet.parameters())
      for i in range(100):
        kr_optim.zero_grad()
        loss = loss_fn(kresnet(self.x), self.y)
        kr_optim.step()

      r_out, kr_out = resnetn(self.x), kresnet(self.x)
      self.assertTrue(torch.all(r_out == kr_out))
