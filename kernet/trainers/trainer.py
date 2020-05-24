"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch

from kernet_future.trainers.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
  def __init__(self, opt, model=None, set_eval=None, optimizer=None,
      val_metric_name='acc', val_metric_obj='max'):
    super(Trainer, self).__init__(opt, model, set_eval, optimizer,
        val_metric_name, val_metric_obj)


  def step(self, input, target, criterion, minimize=True):
    self.model.train()
    if self.set_eval:
      self.set_eval.eval()

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


  def get_eval_output(self, input):
    with torch.no_grad():
      self.model.eval()
      if hasattr(self.model, 'update_centers'):
        return self.model(input, update_centers=False)
      else:
        return self.model(input)
