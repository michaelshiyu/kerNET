"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from kernet.models.resnet import BasicBlock, ResNet


class ResNet18(ResNet):
  def __init__(self, opt):
    super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=opt.n_classes)
