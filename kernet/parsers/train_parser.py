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
        parser.add_argument('--loss', choices=['xe', 'hinge', 'nll'], default='xe',
                            help='The overall loss function to be used. xe is for CrossEntropyLoss, hinge for ' +
                            'MultiMarginLoss (multi-class hinge loss), nll is for NLLLoss (CrossEntropyLoss w/o logsoftmax).')
        parser.add_argument('--shuffle', type=utils.str2bool,
                            nargs='?', const=True, default=True,
                            help='Whether to shuffle data across training epochs. "True" or "t" will be parsed as True (bool); "False" or "f" as False. Same works for all params of bool type.')
        parser.add_argument('--augment_data', type=utils.str2bool,
                            nargs='?', const=True, default=True,
                            help='If True, augment training data. See datasets/__init__.py for the specific augmentations used.')
        parser.add_argument('--train_subset_indices', type=str,
                            help='Path to saved training subset indices, if available.')
        parser.add_argument('--print_freq', type=int, default=100,
                            help='Print training statistics once every this many mini-batches.')
        parser.add_argument('--n_classes', type=int,
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
        parser.add_argument('--max_trainset_size', type=int, default=int(1e12),
                            help='Max size for the training set.')
        parser.add_argument('--always_save', type=utils.str2bool,
                            nargs='?', const=True, default=False,
                            help='If True, always save model after each epoch. Otherwise, save according to validation metric.')
        parser.add_argument('--n_val', type=int, default=100,
                            help='Validation set size. The validation set will be taken from a randomly permuted training set. Can set to 0 if do not need a validation set.')
        parser.add_argument('--dataset_rand_idx', type=str,
                            help='Path to saved permuted dataset indices (used for selecting validation set), if available.')
        parser.add_argument('--max_ori_trainset_size', type=int, default=int(1e12),
                            help='Sample size in the first (optional) random sampling on training set. ' +
                            'This sampling is done before any other processing on the training set such ' +
                            'as train/val split.')
        parser.add_argument('--ori_train_subset_indices', type=str,
                            help='Path to saved subset indices for the first random sampling, if available.')
        parser.add_argument('--ori_balanced', type=utils.str2bool,
                            nargs='?', const=True, default=False,
                            help='If True, the first random sampling will try to sample an equal number of examples ' +
                            'from each class.')

        self.is_train = True
        return parser
