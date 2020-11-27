"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import time
import logging

import torch

import kernet.utils as utils


logger = logging.getLogger()


def train(opt, n_epochs, trainer, loader, val_loader, criterion, device):
    logger.info('Starting training...')

    total_epoch = trainer.start_epoch + n_epochs
    total_batch = len(loader)

    for epoch in range(trainer.start_epoch, total_epoch):
        running_loss = 0.
        running_acc = 0.
        running_time = 0.
        total_loss = 0.
        total_acc = 0.
        total_time = 0.
        if opt.loglevel.upper() not in ['DEBUG', 'NOTSET']:
            # for DEBUG or below, training metrics will be printed in a more verbose fashion
            # instead of in the form of a progress bar
            pbar = utils.ProgressBar(len(loader))
        for batch, (input, target) in enumerate(loader):

            # train step
            start = time.time()
            input, target = \
                input.to(device, non_blocking=True), target.to(
                    device, non_blocking=True)
            output, loss = trainer.step(input, target, criterion)
            end = time.time()

            # get some batch statistics
            _, predicted = torch.max(output, 1)
            acc = 100 * \
                (predicted == target).sum().to(
                    torch.float).item() / target.size(0)
            trainer.log_loss_values({
                f'train_batch_{criterion.__class__.__name__.lower()}': loss,
                'train_batch_acc': acc
            })

            # print statistics
            running_loss += loss
            running_acc += acc
            running_time += end - start
            total_loss += loss
            total_acc += acc
            total_time += end - start
            if batch % opt.print_freq == opt.print_freq - 1:
                message = (
                    f'[epoch: {epoch + 1:5d}/{total_epoch}, batch: {batch + 1:5d}/{total_batch}] '
                    f'{criterion.__class__.__name__.lower()}: {running_loss / opt.print_freq:.3f}, '
                    f'acc (%): {running_acc / opt.print_freq:.3f}, '
                    f'runtime (s): {running_time / opt.print_freq:.3f}'
                )
                logger.debug(message)
                running_loss = 0.
                running_acc = 0.
                running_time = 0.
            if opt.loglevel.upper() not in ['DEBUG', 'NOTSET']:
                message = (
                    f'[epoch: {epoch + 1:5d}/{total_epoch}] '
                    f'avg {criterion.__class__.__name__.lower()}: {total_loss / (batch + 1):.3f}, '
                    f'avg acc (%): {total_acc / (batch + 1):.3f}, '
                    f'avg runtime (s): {total_time / (batch + 1):.3f}'
                )
                pbar.update(message)

        # validate
        if epoch % opt.val_freq == opt.val_freq - 1:
            if val_loader is not None:
                if hasattr(trainer, 'update_centers_eval'):
                    trainer.update_centers_eval()
                correct, total = 0, 0
                total_batch_val = len(val_loader)
                if opt.loglevel.upper() not in ['DEBUG', 'NOTSET']:
                    val_pbar = utils.ProgressBar(len(val_loader))
                for i, (input, target) in enumerate(val_loader):
                    input, target = \
                        input.to(device, non_blocking=True), target.to(
                            device, non_blocking=True)
                    output = trainer.get_eval_output(input)

                    _, predicted = torch.max(output, 1)
                    batch_correct = (predicted == target).sum().item()
                    correct += batch_correct
                    total += target.size(0)

                    if opt.loglevel.upper() not in ['DEBUG', 'NOTSET']:
                        latest_acc = 100 * correct / total
                        message = f'avg val acc (%): {latest_acc:.3f}'
                        val_pbar.update(message)
                    batch_acc = 100 * \
                        batch_correct / target.size(0)
                    message = f'batch: {i + 1}/{total_batch_val}, batch val acc (%): {batch_acc:.3f}'
                    logger.debug(message)

                acc = 100 * correct / total
                trainer.log_loss_values({
                    'val_acc': acc
                })

                message = f'[epoch: {epoch + 1}] val acc (%): {acc:.3f}'
                logger.info(message)
                if opt.schedule_lr:
                    trainer.scheduler_step(acc)
                trainer.save(epoch, acc, force_save=opt.always_save)
            else:
                trainer.save(epoch, 0, force_save=True)
    logger.info('Training finished!')
