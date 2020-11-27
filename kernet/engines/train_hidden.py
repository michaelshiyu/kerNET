"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import time
import logging

import kernet.utils as utils

logger = logging.getLogger()


# TODO if not all batches are of the same size, some of the printed statistics might be slightly
# wrong due to the fact that the final batch might be smaller than the others but stats from all batches
# are sometimes averaged upon printing. The same goes for other scripts in engine/. This does
# not affect training result and all validation accuracy stats are correct

def train_hidden(opt, n_epochs, trainer, loader, val_loader, criterion, part_id, device):
    logger.info(f'Starting training part {part_id}...')

    total_epoch = trainer.start_epoch + n_epochs
    total_batch = len(loader)

    for epoch in range(trainer.start_epoch, total_epoch):
        running_loss = 0.
        running_time = 0.
        total_loss = 0.
        total_time = 0.
        if opt.loglevel.upper() not in ['DEBUG', 'NOTSET']:
            pbar = utils.ProgressBar(len(loader))
        for batch, (input, target) in enumerate(loader):
            start = time.time()

            # train step
            input, target = \
                input.to(device, non_blocking=True), target.to(
                    device, non_blocking=True)
            output, loss = trainer.step(
                input, target, criterion, minimize=False)
            end = time.time()

            # get some batch statistics
            trainer.log_loss_values({
                f'train_{part_id}_batch_{opt.hidden_objective}': loss,
            })

            # print statistics
            running_loss += loss  # revert the sign for loss logging
            running_time += end - start
            total_loss += loss  # revert the sign for loss logging
            total_time += end - start
            if batch % opt.print_freq == opt.print_freq - 1:
                message = (
                    f'[part: {part_id}, epoch: {epoch+1:5d}/{total_epoch}, batch: {batch+1:5d}/{total_batch}] '
                    f'{opt.hidden_objective}: {running_loss/opt.print_freq:.3f}, '
                    f'runtime (s): {running_time/opt.print_freq:.3f}'
                )
                logger.debug(message)
                running_loss = 0.
                running_time = 0.
            if opt.loglevel.upper() not in ['DEBUG', 'NOTSET']:
                message = (
                    f'[part: {part_id}, epoch: {epoch+1:5d}/{total_epoch}] '
                    f'avg {opt.hidden_objective}: {total_loss/(batch+1):.3f}, '
                    f'avg runtime (s): {total_time/(batch+1):.3f}'
                )
                pbar.update(message)

        # validate
        if epoch % opt.val_freq == opt.val_freq - 1:
            if val_loader is not None:
                hidden_obj, total = 0, 0
                total_batch_val = len(val_loader)
                if opt.loglevel.upper() not in ['DEBUG', 'NOTSET']:
                    val_pbar = utils.ProgressBar(len(val_loader))
                for i, (input, target) in enumerate(val_loader):
                    input, target = \
                        input.to(device, non_blocking=True), target.to(
                            device, non_blocking=True)
                    output = trainer.get_eval_output(input)
                    batch_obj = criterion(output, target).item()
                    hidden_obj += batch_obj
                    total += 1

                    if opt.loglevel.upper() not in ['DEBUG', 'NOTSET']:
                        latest_obj = hidden_obj / total
                        message = f'avg val {opt.hidden_objective}:  {latest_obj:.3f}'
                        val_pbar.update(message)
                    message = f'batch: {i + 1}/{total_batch_val}, batch val {opt.hidden_objective}: {batch_obj:.3f}'
                    logger.debug(message)

                hidden_obj /= total
                trainer.log_loss_values({
                    f'val_{part_id}_{opt.hidden_objective}': hidden_obj
                })
                message = f'[part {part_id}, epoch: {epoch+1}] val {opt.hidden_objective}: {hidden_obj:.3f}'
                logger.info(message)
                if opt.schedule_lr:
                    trainer.scheduler_step(hidden_obj)
                trainer.save(epoch, hidden_obj,
                             model_name=f'net_part{part_id}.pth', force_save=opt.always_save)
            else:
                trainer.save(epoch, 0,
                             model_name=f'net_part{part_id}.pth', force_save=True)

    logger.info(f'Part {part_id} training finished!')

    # load best part for the next training session
    logger.info(
        'Before proceeding, load saved checkpoint for training the next part...')
    tmp, opt.checkpoint_dir = opt.checkpoint_dir, opt.save_dir
    trainer.load(f'net_part{part_id}.pth')
    opt.checkpoint_dir = tmp
