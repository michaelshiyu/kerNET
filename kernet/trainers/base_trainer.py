"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import logging

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


logger = logging.getLogger()


class BaseTrainer(torch.nn.Module):
    def __init__(self, opt, model=None, set_eval=None, optimizer=None,
                 val_metric_name='acc', val_metric_obj='max'):
        """
        Args:
          set_eval: Part of the model to be set to eval during training
          val_metric_name (str): The name of the validation metric. Just 
            some helpful information to be logged. Can be any name. Default: 'acc'.
          val_metric_obj (str): Whether to minimize or maximize the validation
            metric. One of 'min' or 'max'. Default: 'max'.
        """
        super(BaseTrainer, self).__init__()
        if val_metric_obj not in ['min', 'max']:
            raise ValueError()
        self.opt = opt
        self.steps_taken = 0  # the total number of train steps taken
        self.start_epoch = 0
        self.val_metric_name = val_metric_name
        self.val_metric_obj = val_metric_obj
        self.best_val_metric = float(
            'inf') if val_metric_obj == 'min' else -float('inf')
        self.model = model
        self.set_eval = set_eval

        if opt.is_train:
            self.optimizer = optimizer
            if opt.schedule_lr:
                self.scheduler = ReduceLROnPlateau(self.optimizer, val_metric_obj,
                                                   factor=opt.lr_schedule_factor,
                                                   patience=opt.lr_schedule_patience, verbose=True)

    def load(self, model_name='net.pth'):
        filename = os.path.join(self.opt.checkpoint_dir, model_name)
        if not os.path.isfile(filename):
            logger.warning('Did not load {} since {} does not exist.'.format(
                model_name, filename))
            return

        logger.info('Loading checkpoint from {}...'.format(filename))
        checkpoint = torch.load(filename)

        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError:
            own_keys = set(self.model.state_dict().keys())
            load_keys = set(checkpoint['state_dict'].keys())
            missing = own_keys - load_keys
            extra = load_keys - own_keys
            if missing:
                logger.warning(
                    'Missing keys from model checkpoint:\n' + ', '.join(sorted(missing)))
            if extra:
                logger.warning(
                    'Extra keys from model checkpoint:\n' + ', '.join(sorted(extra)))
            logger.warning(
                'Loading only the matched keys from model checkpoint...')
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        if self.opt.is_train:
            if self.opt.schedule_lr:  # TODO potentially overrides new params
                try:
                    self.scheduler.load_state_dict(
                        checkpoint['scheduler_state_dict'])
                except KeyError:
                    # this is useful in cases where the saved model did not use lr scheduling but one
                    # wants to schedule lr for fine-tuning
                    logger.warning(
                        'Did not find scheduler state dict from checkpoint. Not loading it...')

            self.best_val_metric = checkpoint['best_val_metric']
            self.steps_taken = checkpoint['steps_taken']
            # start from the next epoch of the previously saved epoch as we only save
            # at the end of an epoch
            self.start_epoch = checkpoint['epoch'] + 1
            logger.debug('Current best val metric ({}): {:.3f}.'.format(
                self.val_metric_name, self.best_val_metric))
            # all epoch numbers are 0-indexed
            logger.debug('Now starting from epoch {}...'.format(
                self.start_epoch+1))
        logger.info('Checkpoint loaded!')

    def save(self, epoch, val_metric_value, model_name='net.pth', force_save=False):
        if not force_save:
            if self.val_metric_obj == 'min':
                if val_metric_value >= self.best_val_metric:
                    return
            else:
                if val_metric_value <= self.best_val_metric:
                    return

        self.best_val_metric = val_metric_value
        save_file = os.path.join(self.opt.save_dir, model_name)
        logger.info('Saving checkpoint to {}...'.format(save_file))

        states = {
            'epoch': epoch,
            'steps_taken': self.steps_taken,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric
        }

        if self.opt.schedule_lr:
            states['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(states, save_file)
        logger.info('Checkpoint saved!')

    def step(self, input, target, criterion, minimize=True):
        """
        Sets model to train mode, then perform one training step.

        Returns:
          output: Model output (without backward graph).
          loss: Loss value (without backward graph).
          minimize (bool): Whether to minimize or maximize the criterion.
            Default: True.
        """
        raise NotImplementedError()

    def get_eval_output(self, input):
        """
        Sets model to eval mode, then evaluates the output from input
        without tracking the backward graph.

        Returns:
          output: Model output (without backward graph).
        """
        raise NotImplementedError()

    def scheduler_step(self, val_loss_value):
        self.scheduler.step(val_loss_value)

    def log_loss_values(self, loss_dict):
        for k, v in loss_dict.items():
            logger.add_scalar(k, v, self.steps_taken)
