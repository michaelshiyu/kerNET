"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging

import torch

import kernet.utils as utils
from kernet.layers.klinear import _kLayer, kLinear
from kernet.models import Flatten
from kernet.models.base_model import BaseModel


logger = logging.getLogger()


# TODO support Gaussian kernel (need to deal with centers and convert to committee if necessary)
# TODO support splitting into more parts
# TODO right now all activation vectors are normalized to unit norm, which needs not be the case
class kMLP(BaseModel):
    def __init__(self, opt, *args, **kwargs):
        super(kMLP, self).__init__()

        if opt.activation == 'tanh':
            self.kernel = 'nn_tanh'
            self.evaluation = 'direct'
        elif opt.activation == 'sigmoid':
            self.kernel = 'nn_sigmoid'
            self.evaluation = 'direct'
        elif opt.activation == 'relu':
            self.kernel = 'nn_relu'
            self.evaluation = 'direct'
        else:
            raise NotImplementedError()

        self.flatten = Flatten()
        arch_list = [int(_) for _ in opt.arch.split('_')]
        for i in range(len(arch_list) - 1):
            if i == 0:
                setattr(
                    self, f'layer_{i+1}',
                    torch.nn.Linear(
                        in_features=arch_list[i], out_features=arch_list[i+1])
                )
            else:
                setattr(
                    self, f'layer_{i+1}',
                    kLinear(in_features=arch_list[i], out_features=arch_list[i+1],
                            kernel=self.kernel, evaluation=self.evaluation)
                )
        self.opt = opt
        self.arch_list = arch_list
        self.n_layers = len(arch_list) - 1
        self.print_network(self)

    def forward(self, input, update_centers=True):
        output = self.flatten(input)  # flatten input
        for i in range(self.n_layers):
            layer = getattr(self, f'layer_{i+1}')
            output = layer(output)
        return output

    def split(self, n_parts, **kwargs):
        """
        Split the model into n_parts parts for modular training.
        The last part is always the (kernelized) output layer.
        The ith trainable module will always be a single-layer one for i > 1.

        Args:
          n_parts (int): The number of parts to split.

        Returns:
          models (tuple): A tuple of n_parts models.
          trainables (tuple): A tuple of n_parts sets of trainable params.
        """
        if n_parts < 1:
            raise ValueError(
                f'n_parts should be at least 1, got {n_parts} instead.')
        if n_parts > self.n_layers:
            logger.warning(
                f'{self.__class__.__name__} can be split into at most {self.n_layers} parts. '
                f'Splitting it into {self.n_layers} parts instead of the requested {n_parts}...'
            )
            n_parts = self.n_layers

        if n_parts == 1:
            models, trainables = (self,), (self,)
        else:
            models, trainables = [], []
            for i in range(n_parts - 1, 0, -1):
                module = utils.attach_head(torch.nn.Sequential(
                    *list(self.children())[:-i]), self.opt)
                models.append(module)
                trainables.append(module[-1] if i != n_parts - 1 else module)
            output_layer = list(self.children())[-1]
            models.append(self)
            trainables.append(output_layer)

        logger.debug('Splitting {} into:'.format(self.__class__.__name__))
        for i, t in enumerate(trainables):
            logger.debug('part {}:\n'.format(i + 1) + str(t))

        return models, [_.parameters() for _ in trainables]

    @staticmethod
    def modify_commandline_options(parser, **kwargs):
        parser = _kLayer.modify_commandline_options(parser, **kwargs)
        parser.add_argument('--arch', type=str, default='784_1024_2048_512_128_10',
                            help='The number of layers and the layer widths of the MLP. ' +
                                 'Should be a "_"-separated string of integers, ' +
                                 'each pair of numbers specifying a layer shape.'
                            )
        return parser
