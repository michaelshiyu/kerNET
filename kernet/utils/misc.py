"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import sys
import glob
import time
import logging
import argparse
import functools
from shutil import get_terminal_size

import torch
import torchvision

logger = logging.getLogger()
INF = float('inf')
AVOID_ZERO_DIV = torch.tensor(1e-12)


def examine_checkpoints(dirs: str) -> dict:
    """
    Given checkpoint directories, searches all checkpoints and returns a dict of
    {directory1 name: {checkpoint1 name: best validation metric, checkpoint2 name: ...}, ...}.

    Wildcards are supported for the argument dirs.
    """
    dirs = glob.glob(dirs)

    if len(dirs) == 0:
        return {}
    if len(dirs) > 1:
        results = {}
        for checkpoint_dir in dirs:
            results = {**results, **examine_checkpoints(checkpoint_dir)}
        return results

    checkpoint_dir = dirs[0]
    print(f'processing {checkpoint_dir}...')
    checkpoint_names = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    info = {}
    for checkpoint_name in checkpoint_names:
        checkpoint = torch.load(checkpoint_name)
        info[os.path.splitext(os.path.basename(checkpoint_name))[
            0]] = checkpoint['best_val_metric']
    return {os.path.basename(checkpoint_dir): info}


def mask_loss_fn(mask_val):
    # only computes loss on examples where target == mask_val
    def wrapper(loss_fn):
        @functools.wraps(loss_fn)
        def wrapped(input, target):
            idx = target == mask_val
            return loss_fn(input[idx], target[idx])
        return wrapped
    return wrapper


def one_hot_encode(target, n_classes):
    """
    One-hot encode. Uses the values in target as the positions of
    the 1s in the resulting one-hot vectors.

    Args:
      target (tensor): Categorical labels with values in
        {0, 1, ..., n_classes-1}.
      n_classes (int)

    Shape:
      - Input: (n_examples, 1) or (n_examples,)
      - Output: (n_examples, n_classes)
    """
    if len(target.size()) > 1:
        target.squeeze_()
    target_onehot = target.new_zeros(target.size(0), n_classes)
    target_onehot[range(target.size(0)), target] = 1
    return target_onehot


def to_unit_vector(input):
    """
    Make each vector in input have Euclidean norm 1.

    Shape:
      input: (n_examples, d)
    """
    return input / torch.max(
        torch.norm(input, dim=1, keepdim=True),
        AVOID_ZERO_DIV.to(input.device)
    )


def get_optimizer(opt, params, lr, weight_decay, **kwargs):
    """
    Returns an optimizer according to opt.
    """
    logger.debug('Getting optimizer {} w/ lr: {}, weight decay: {}, '.format(opt.optimizer, lr, weight_decay) +
                 ', '.join(str(k) + ': ' + str(v) for k, v in kwargs.items()))
    if opt.optimizer == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=kwargs['momentum'])


def sample(tensor, n):
    """
    Returns a random sample of size n from tensor.

    tensor may have shape (N, ...). And the first dimension is assumed
    to be the batch dimension
    """
    logger.debug('Sampling {} from a tensor of size {}...'.format(
        n, list(tensor.size())))
    perm = torch.randperm(len(tensor))
    idx = perm[:n]
    return tensor[idx]


def supervised_sample(tensor, labels, n: int, return_labels=False, indices_only=False):
    """
    Returns a random sample of size n from tensor.
    The sample will consist of an equal number of examples from each class.
    The number of classes is inferred from labels (the number of unique
    labels in labels).

    tensor and labels should both be torch tensors.
    tensor may have shape (N, ...). And the first dimension is assumed
    to be the batch dimension.
    labels should be a vector (1d tensor) of N scalar labels.

    If return_labels is True, return the sampled subtensor as well as the labels.
    Otherwise, return only the subtensor.

    If indices_only is True, returns only the sampled indices in the given dataset. In this case, 
    return_labels has no effect.
    """
    ulabels = torch.unique(labels)
    if n % len(ulabels):
        raise ValueError(
            'Please select n to be divisible by the number of classes')
    if n < len(ulabels):
        raise ValueError(
            'Please select n to be higher than the number of classes')

    logger.debug(
        'Sampling {} from a tensor of size {}, with an equal number of examples from each of the {} classes...'.format(
            n, list(tensor.size()), len(ulabels)))

    nn = n // len(ulabels)
    for i, l in enumerate(ulabels):
        l_idx = torch.where(labels == l)[0]
        if i == 0:
            # shuffle before sampling
            idx = l_idx[torch.randperm(len(l_idx))[:nn]]
        else:
            idx = torch.cat((idx, l_idx[torch.randperm(len(l_idx))[:nn]]))

    if len(idx) < n:
        logger.warning('Sampled {} examples instead of {}. '.format(len(idx), n) +
                       'May have been caused by some class(es) not having enough examples to sample from.')

    # shuffle again. Otherwise examples are ordered by classes
    idx = idx[torch.randperm(len(idx))]
    if indices_only:
        # TODO in this mode, tensor is not needed as an arg
        return idx
    else:
        if return_labels:
            return tensor[idx], labels[idx]
        else:
            return tensor[idx]


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('true', 't'):
        return True
    elif s.lower() in ('false', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            'Cannot interpret {} as bool'.format(s))


def make_deterministic(seed):
    import os
    import random
    import numpy as np
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def upper_tri(mtrx):
    """
    Returns elements on the upper triangle minus the main diagonal of mtrx.

    Args:
      mtrx (2D matrix): The matrix to be processed.

    Returns a 1D vector of elements from mtrx.
    """
    upper_tri_indices = torch.triu_indices(
        mtrx.size(0), mtrx.size(1), offset=1)
    return mtrx[upper_tri_indices[0], upper_tri_indices[1]]


class ProgressBar:
    """
    A progress bar that can print the progress.

    Reference:
    https://github.com/xinntao/BasicSR
    """

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (
            bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print(f'terminal width is too small ({terminal_width}), '
                  'please consider widen the terminal for better '
                  'progressbar visualization')
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write(f"[{' ' * self.bar_width}] 0/{self.task_num}, "
                             f'elapsed: 0s, ETA:\nStart...\n')
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time + 1e-8
        b_per_s = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write(
                '\033[J'
            )  # clean the output (remove extra chars since last display)
            sys.stdout.write(
                f'[{bar_chars}] {self.completed}/{self.task_num}, '
                f'{b_per_s:.1f} batches/s, elapsed: {int(elapsed + 0.5)}s, '
                f'ETA: {eta:5}s\n{msg}\n')
        else:
            sys.stdout.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                f' {b_per_s:.1f} batches/s')
        sys.stdout.flush()
