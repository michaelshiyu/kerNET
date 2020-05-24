"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import kernet.utils as utils
from kernet.parsers.base_parser import BaseParser

class TrainParser(BaseParser):
  def initialize(self, parser):
    super().initialize(parser)

    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam',
                        help='The optimizer to be used.')
    parser.add_argument('--shuffle', type=utils.str2bool,
                        nargs='?', const=True, default=True,
      help='Whether to shuffle data across training epochs. "True" or "t" will be parsed as True (bool); "False" or "f" as False. Same works for all params of bool type.')
    parser.add_argument('--augment_data', type=utils.str2bool,
                        nargs='?', const=True, default=True,
                        help='If True, augment training data. See datasets/__init__.py for the specific augmentations used.')
    parser.add_argument('--print_freq', type=int, default=100,
      help='Print training statistics once every this many mini-batches.')
    parser.add_argument('--n_classes', type=int, default=10,
      help='The number of classes in the data.')
    parser.add_argument('--seed', type=int,
      help='Random seed. If specified, training will be (mostly) deterministic.')
    parser.add_argument('--tf_log', type=utils.str2bool,
                        nargs='?', const=True, default=True,
      help='Whether to log training statistics with tensorboard.')
    parser.add_argument('--schedule_lr', type=utils.str2bool,
                        nargs='?', const=True, default=False,
      help='Whether to schedule learning rate with the ReduceLROnPlateau scheduler.')
    parser.add_argument('--lr_schedule_factor', type=float, default=.1,
      help='The factor argument passed to the scheduler.')
    parser.add_argument('--lr_schedule_patience', type=int, default=10,
      help='The patience argument passed to the scheduler.')
    parser.add_argument('--val_freq', type=int, default=1,
      help='Validate once every this many epochs.')
    parser.add_argument('--adversarial', type=utils.str2bool,
                        nargs='?', const=True, default=False,
      help='If specified, use adversarial training with a PGD adversary.')
    parser.add_argument('--pgd_eps', type=float, default=0.3,
      help='Total epsilon for PGD attack.')
    parser.add_argument('--pgd_norm', type=str, default='inf', choices=['inf', '2'],
      help='Norm for PGD attack.')
    parser.add_argument('--pgd_step_eps', type=float, default=0.01,
      help='Per-step epsilon for PGD attack.')
    parser.add_argument('--pgd_n_steps', type=int, default=50,
      help='Number of attack steps for PGD attack.')
    parser.add_argument('--max_trainset_size', type=int, default=int(1e12),
      help='Max size for the training set.')

    self.is_train = True
    return parser
