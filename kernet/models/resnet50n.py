"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from kernet.models.resnetn import ResNetN
from kernet.models.resnet import Bottleneck


class ResNet50N(ResNetN):
    def __init__(self, opt):
        super(ResNet50N, self).__init__(Bottleneck, [
            3, 4, 6, 3], in_channels=opt.in_channels, num_classes=opt.n_classes)
