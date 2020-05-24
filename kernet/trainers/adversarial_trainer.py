"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging

import numpy as np

import torch

from cleverhans.future.torch.utils import optimize_linear
from cleverhans.future.torch.utils import clip_eta
from kernet.trainers.trainer import Trainer


logger = logging.getLogger()

class AdversarialTrainer(Trainer):
  def __init__(self, opt, model=None, set_eval=None, optimizer=None,
      val_metric_name='acc', val_metric_obj='max'):
    super(AdversarialTrainer, self).__init__(opt, model, set_eval, optimizer,
        val_metric_name, val_metric_obj)


  def step(self, input, target, criterion, minimize=True):
    self.model.train()
    if self.set_eval:
      self.set_eval.eval()

    # replace clean examples with adversarial examples for adversarial training
    # FIXME when training the hidden layers, untargeted attack would cause a bug
    # since one cannot get the model to predict labels (the model here does not have
    # an output layer. Also, if we use the untrained output layer to get predicted
    # labels, the hidden layers would be trained w/ different adversarial examples
    # than the output layer)

    # targeted attack w/ classes shifted by 1 (mod n_classes)
    attack_target = target.clone()
    attack_target[attack_target == (self.opt.n_classes - 1)] = -1
    attack_target += 1

    logger.debug('PGD attack starts...')
    logger.debug('Attacker params: norm {}, eps {}, eps per step {}, n_steps {}'.format(
      self.opt.pgd_norm, self.opt.pgd_eps, self.opt.pgd_step_eps, self.opt.pgd_n_steps
    ))
    input = projected_gradient_descent(self.model, input, self.opt.pgd_eps,
        self.opt.pgd_step_eps, self.opt.pgd_n_steps, self.opt.pgd_norm, criterion, 
        minimize=minimize, targeted=True, y=attack_target)
    logger.debug('PGD attack finished!')

    self.optimizer.zero_grad()
    output = self.model(input)

    if minimize:
      loss = criterion(output, target)
    else:
      loss = -criterion(output, target)

    loss.backward()
    self.optimizer.step()
    self.steps_taken += 1
    
    # on why use .detach() but not .data:
    # https://pytorch.org/blog/pytorch-0_4_0-migration-guide/
    if minimize:
      return output.detach(), loss.item()
    else:
      return output.detach(), -loss.item()


def fast_gradient_method(model_fn, x, eps, norm, criterion, minimize=True,
                         clip_min=None, clip_max=None, y=None, targeted=False, sanity_checks=False):
  """
  Modified from the fast_gradient_method in cleverhans.
  Can be used to attack under any criterion function.

  PyTorch implementation of the Fast Gradient Method.
  :param model_fn: a callable that takes an input tensor and returns the model logits.
  :param x: input tensor.
  :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
  :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
  :param criterion: The criterion function used by the learner.
  :param minimize: The objective of the learner (not the attacker!): Whether to minimize or 
            maximize the criterion.
  :param clip_min: (optional) float. Minimum float value for adversarial example components.
  :param clip_max: (optional) float. Maximum float value for adversarial example components.
  :param y: (optional) Tensor with true labels. If targeted is true, then provide the
            target label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting adversarial samples. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
  :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
            memory or for unit tests that intentionally pass strange input)
  :return: a tensor for the adversarial example
  """
  if norm not in [np.inf, 1, 2]:
    raise ValueError("Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm))
  if eps < 0:
    raise ValueError("eps must be greater than or equal to 0, got {} instead".format(eps))
  if eps == 0:
    return x
  if clip_min is not None and clip_max is not None:
    if clip_min > clip_max:
      raise ValueError(
          "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
              clip_min, clip_max))

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    assert_ge = torch.all(torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype)))
    asserts.append(assert_ge)

  if clip_max is not None:
    assert_le = torch.all(torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype)))
    asserts.append(assert_le)

  # x needs to be a leaf variable, of floating point type and have requires_grad being True for
  # its grad to be computed and stored properly in a backward call
  x = x.clone().detach().to(torch.float).requires_grad_(True)

  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    _, y = torch.max(model_fn(x), 1)

  # Compute loss
  loss = criterion(model_fn(x), y)
  # If the objective of the learner is to minimize loss, the attacker should try to maxmize it
  if minimize:
    loss = -loss
  # If attack is targeted, the attacker should minimize/maximize in the same 
  # way as the learner does
  if targeted:
    loss = -loss

  # Define gradient of loss wrt input
  loss.backward()
  optimal_perturbation = optimize_linear(x.grad, eps, norm)

  # Add perturbation to original example to obtain adversarial example
  adv_x = x + optimal_perturbation

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    if clip_min is None or clip_max is None:
      raise ValueError(
          "One of clip_min and clip_max is None but we don't currently support one-sided clipping")
    adv_x = torch.clamp(adv_x, clip_min, clip_max)

  if sanity_checks:
    assert np.all(asserts)
  return adv_x


def projected_gradient_descent(model_fn, x, eps, eps_iter, nb_iter, norm, criterion, minimize=True,
                               clip_min=None, clip_max=None, y=None, targeted=False,
                               rand_init=True, rand_minmax=None, sanity_checks=True):
  """
  Modified from the fast_gradient_method in cleverhans.
  Can be used to attack under any criterion function.

  This class implements either the Basic Iterative Method
  (Kurakin et al. 2016) when rand_init is set to False. or the
  Madry et al. (2017) method if rand_init is set to True.
  Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
  Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
  :param model_fn: a callable that takes an input tensor and returns the model logits.
  :param x: input tensor.
  :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
  :param eps_iter: step size for each attack iteration
  :param nb_iter: Number of attack iterations.
  :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
  :param criterion: The criterion function used by the learner.
  :param minimize: The objective of the learner (not the attacker!): Whether to minimize or 
            maximize the criterion.
  :param clip_min: (optional) float. Minimum float value for adversarial example components.
  :param clip_max: (optional) float. Maximum float value for adversarial example components.
  :param y: (optional) Tensor with true labels. If targeted is true, then provide the
            target label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting adversarial samples. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
  :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
  :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
            which the random perturbation on x was drawn. Effective only when rand_init is
            True. Default equals to eps.
  :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
            memory or for unit tests that intentionally pass strange input)
  :return: a tensor for the adversarial example
  """
  if norm == 1:
    raise NotImplementedError("It's not clear that FGM is a good inner loop"
                              " step for PGD when norm=1, because norm=1 FGM "
                              " changes only one pixel at a time. We need "
                              " to rigorously test a strong norm=1 PGD "
                              "before enabling this feature.")
  if norm not in [np.inf, 2]:
    raise ValueError("Norm order must be either np.inf or 2.")
  if eps < 0:
    raise ValueError(
        "eps must be greater than or equal to 0, got {} instead".format(eps))
  if eps == 0:
    return x
  if eps_iter < 0:
    raise ValueError(
        "eps_iter must be greater than or equal to 0, got {} instead".format(eps_iter))
  if eps_iter == 0:
    return x

  assert eps_iter <= eps, (eps_iter, eps)
  if clip_min is not None and clip_max is not None:
    if clip_min > clip_max:
      raise ValueError(
          "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
              clip_min, clip_max))

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    assert_ge = torch.all(torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype)))
    asserts.append(assert_ge)

  if clip_max is not None:
    assert_le = torch.all(torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype)))
    asserts.append(assert_le)

  # Initialize loop variables
  if rand_init:
    if rand_minmax is None:
      rand_minmax = eps
    eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
  else:
    eta = torch.zeros_like(x)

  # Clip eta
  eta = clip_eta(eta, norm, eps)
  adv_x = x + eta
  if clip_min is not None or clip_max is not None:
    adv_x = torch.clamp(adv_x, clip_min, clip_max)

  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    _, y = torch.max(model_fn(x), 1)

  i = 0
  while i < nb_iter:
    adv_x = fast_gradient_method(model_fn, adv_x, eps_iter, norm, criterion, minimize,
                                 clip_min=clip_min, clip_max=clip_max, y=y, targeted=targeted)

    # Clipping perturbation eta to norm norm ball
    eta = adv_x - x
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta

    # Redo the clipping.
    # FGM already did it, but subtracting and re-adding eta can add some
    # small numerical error.
    if clip_min is not None or clip_max is not None:
      adv_x = torch.clamp(adv_x, clip_min, clip_max)
    i += 1

  asserts.append(eps_iter <= eps)
  if norm == np.inf and clip_min is not None:
    # TODO necessary to cast clip_min and clip_max to x.dtype?
    asserts.append(eps + clip_min <= clip_max)

  if sanity_checks:
    assert np.all(asserts)
  return adv_x
