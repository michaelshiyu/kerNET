"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from kernet_future.models.resnetn import ResNetN
from kernet_future.models.resnet import BasicBlock


class ResNet34N(ResNetN):
  def __init__(self, opt):
    super(ResNet34N, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=opt.n_classes)
