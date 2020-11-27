"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from kernet.models.kresnet import kResNet
from kernet.models.resnet import BasicBlock


class kResNet34(kResNet):
    def __init__(self, opt, centers=None):
        super(kResNet34, self).__init__(opt, centers,
                                        block=BasicBlock, num_blocks=[
                                            3, 4, 6, 3],
                                        in_channels=opt.in_channels, num_classes=opt.n_classes)
