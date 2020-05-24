"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import unittest

import numpy as np

import torch

import kernet_future.utils as utils


class UtilsTest(unittest.TestCase):
  # NOTE the kernel functions are tested with kcore module in 
  # test_kcore.py

  def setUp(self):
    # allow tests to be run on GPU if possible
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


  def test_one_hot_encode_target1(self):
    target1 = torch.tensor([1, 3, 0]).to(self.device)
    target2 = torch.tensor([[1], [3], [0]]).to(self.device)
    n_classes = 4

    for target in [target1, target2]:
      target_onehot = utils.one_hot_encode(target, n_classes=n_classes)
      self.assertTrue(np.all(np.equal(
        target_onehot.cpu().numpy(),
        np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
        )))
