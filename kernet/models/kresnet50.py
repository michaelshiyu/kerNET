"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from kernet.models.kresnet import kResNet
from kernet.models.resnet import Bottleneck


class kResNet50(kResNet):
  def __init__(self, opt, centers=None):
    super(kResNet50, self).__init__(opt, centers,
      block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=opt.n_classes)
