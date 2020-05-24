"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from kernet_future.models.kresnet import kResNet
from kernet_future.models.resnet import BasicBlock


class kResNet18(kResNet):
  def __init__(self, opt, centers=None):
    super(kResNet18, self).__init__(opt, centers,
      block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=opt.n_classes)
