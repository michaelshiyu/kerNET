"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging
import functools

from easydict import EasyDict as edict

import torch
from torch.nn.modules.loss import _Loss

import kernet.utils as utils
import kernet.models as models


logger = logging.getLogger()


class SRSLoss(_Loss):
    """
    The Supervised Representation Similarity (SRS) loss defined as
      (input, target) 
        -> loss(map_target(target), map_input(input)).
    """

    def __init__(self, map_input, map_target, loss):
        """
        Args:
          map_input: A function (executable) that evaluates on every pair of examples
          in input.

          map_target: A function (executable) that evaluates on target. 

          loss: A bivariate loss function (executable) for the outputs from map_input and 
            map_target. Specify 'reduction' when instantiating this object and passing the instance to 
            SRSLoss will make SRSLoss inherit the specified reduction behavior. 
        """
        super(SRSLoss, self).__init__()
        self.map_input = map_input
        self.map_target = map_target
        self.loss = loss

    def forward(self, input, target):
        """
        Args:
          input (tensor): The batch of representations to compare. Shape: (N, d1, d2, ...).
          target (tensor): The target of the batch. Shape: (N, p1, p2, ...).
        """
        res_input = self.map_input(input)
        res_target = self.map_target(target)
        return self.loss(
            res_input.view([-1] + list(res_input.size())[2:]),
            res_target.view([-1] + list(res_target.size())[2:])
        )


# some reference SRS instantiations are given below, all of which are to be maximized
# in all instantiations, pairs of the form (x, x) are not considered

def _make_fn(map_input, map_target, loss):
    """
    A helper method that abstracts some boilerplate code
    """
    return SRSLoss(
        map_input=map_input,
        map_target=map_target,
        loss=loss
    )


def _srs_raw(phi, n_classes, neo):
    """
    In the regular version, the returned loss function computes sum of all kernel
    values from positive pairs minus that of all kernel values from negative pairs
    divided by the total number of pairs.

    In the negative only version, the returned loss function computes -1 times
    the sum of all kernel values from negative pairs divided by the total number
    of negative pairs.

    Args:
      phi: A Phi object.
      n_classes (int): The number of classes in the dataset.
      neo (bool): Whether to use the NEgative Only version, in which only
        the negative pairs are considered.

    Returns an SRS loss function.
    """

    def map_input(x): return torch.where(
        torch.eye(len(x)).to(x.device) == 1,
        torch.tensor(0.).to(x.device),
        phi.get_k_mtrx(x, x)
    )  # removes the main diagonal
    def map_target(x): return phi.get_ideal_k_mtrx(x, x, n_classes=n_classes)

    def loss_fn(input, target):
        if neo:
            mask = torch.where(
                target == phi.k_min,
                torch.tensor(-1., device=input.device),
                torch.tensor(0., device=input.device)
            )
            return torch.sum(input * mask) / (torch.sum(torch.abs(mask)))
        else:
            mask = torch.where(
                target == phi.k_max,
                torch.tensor(1., device=input.device),
                torch.tensor(-1., device=input.device)
            )
            # removes the main diagonal
            return torch.sum(input * mask) / (torch.sum(torch.abs(mask)) - len(input)**.5)

    return _make_fn(
        map_input=map_input,
        map_target=map_target,
        loss=loss_fn,
    )


def _srs_nmse(phi, n_classes, neo):
    """
    Negative MSE between the actual kernel mtrx and the perfect one.
    """

    def map_input(x): return phi.get_k_mtrx(x, x)
    def map_target(x): return phi.get_ideal_k_mtrx(x, x, n_classes=n_classes)
    _loss_fn = torch.nn.MSELoss(reduction='mean')

    if neo:
        @utils.mask_loss_fn(phi.k_min)
        def loss_fn(input, target):
            return -_loss_fn(input, target)
    else:
        def loss_fn(input, target):
            return -_loss_fn(input, target)

    return _make_fn(
        map_input=map_input,
        map_target=map_target,
        loss=loss_fn,
    )


def _srs_alignment(phi, n_classes, neo):
    """
    Cosine similarity between the actual kernel mtrx and the perfect one.
    """

    def map_input(x): return phi.get_k_mtrx(x, x)
    def map_target(x): return phi.get_ideal_k_mtrx(x, x, n_classes=n_classes)
    _loss_fn = torch.nn.CosineSimilarity(dim=0)

    if neo:
        if phi.k_min == 0:
            raise ValueError(
                'srs_alignment_neo is not defined for the given phi.')

        @utils.mask_loss_fn(phi.k_min)
        def loss_fn(input, target):
            return _loss_fn(input, target)
    else:
        loss_fn = _loss_fn

    return _make_fn(
        map_input=map_input,
        map_target=map_target,
        loss=loss_fn,
    )


def _srs_upper_tri_alignment(phi, n_classes, neo):
    """
    Cosine similarity between the actual kernel mtrx and the
    perfect one but only considering the upper triangle minus
    the main diagonal.
    """
    def map_input(x): return utils.upper_tri(phi.get_k_mtrx(x, x))

    def map_target(x): return utils.upper_tri(
        phi.get_ideal_k_mtrx(x, x, n_classes=n_classes))

    _loss_fn = torch.nn.CosineSimilarity(dim=0)

    if neo:
        if phi.k_min == 0:
            raise ValueError(
                'srs_upper_tri_alignment_neo is not defined for the given phi.')

        @utils.mask_loss_fn(phi.k_min)
        def loss_fn(input, target):
            return _loss_fn(input, target)
    else:
        loss_fn = _loss_fn

    return _make_fn(
        map_input=map_input,
        map_target=map_target,
        loss=loss_fn,
    )


def _srs_contrastive(phi, n_classes, neo):
    """
    A contrastive-loss-like instantiation.
    """

    def map_input(x): return torch.where(
        torch.eye(len(x)).to(x.device) == 1,
        torch.tensor(-utils.INF).to(x.device),
        phi.get_k_mtrx(x, x)
    )  # removes the main diagonal
    def map_target(y): return phi.get_ideal_k_mtrx(y, y, n_classes=n_classes)
    if neo:
        def loss_fn(x, y): return -torch.mean(torch.exp(x[y == phi.k_min]))
    else:
        def loss_fn(x, y): return torch.sum(torch.exp(x[y == phi.k_max])) / \
            torch.sum(torch.exp(x))

    return _make_fn(
        map_input=map_input,
        map_target=map_target,
        loss=loss_fn,
    )


def _srs_log_contrastive(phi, n_classes, neo):
    """
    A contrastive-loss-like instantiation.
    """

    if neo:
        def loss_fn(x, y): return - \
            torch.log(torch.mean(torch.exp(x[y == phi.k_min])))
    else:
        def loss_fn(x, y):
            dims = list(range(len(x.size())))
            return torch.logsumexp(x[y == phi.k_max], dim=0) - torch.logsumexp(x, dim=dims)

    def map_input(x): return torch.where(
        torch.eye(len(x)).to(x.device) == 1,
        torch.tensor(-utils.INF).to(x.device),
        phi.get_k_mtrx(x, x)
    )  # removes the main diagonal
    def map_target(y): return phi.get_ideal_k_mtrx(y, y, n_classes=n_classes)

    return _make_fn(
        map_input=map_input,
        map_target=map_target,
        loss=loss_fn,
    )


srs_raw_neo = functools.partial(_srs_raw, neo=True)
srs_raw = functools.partial(_srs_raw, neo=False)
srs_nmse_neo = functools.partial(_srs_nmse, neo=True)
srs_nmse = functools.partial(_srs_nmse, neo=False)
srs_upper_tri_alignment_neo = functools.partial(
    _srs_upper_tri_alignment, neo=True)
srs_upper_tri_alignment = functools.partial(
    _srs_upper_tri_alignment, neo=False)
srs_alignment_neo = functools.partial(_srs_alignment, neo=True)
srs_alignment = functools.partial(_srs_alignment, neo=False)
srs_contrastive_neo = functools.partial(_srs_contrastive, neo=True)
srs_contrastive = functools.partial(_srs_contrastive, neo=False)
srs_log_contrastive_neo = functools.partial(
    _srs_log_contrastive, neo=True)
srs_log_contrastive = functools.partial(
    _srs_log_contrastive, neo=False)
