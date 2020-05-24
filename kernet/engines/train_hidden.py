"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import time
import logging


logger = logging.getLogger()

def train_hidden(opt, n_epochs, trainer, loader, val_loader, criterion, part_id, device):
  logger.info('Starting training part %d...' % part_id)

  total_epoch = trainer.start_epoch + n_epochs
  total_batch = len(loader)

  for epoch in range(trainer.start_epoch, total_epoch):
    running_loss = 0.
    running_time = 0.
    for batch, (input, target) in enumerate(loader):
      start = time.time()

      # train step
      input, target = \
        input.to(device, non_blocking=True), target.to(device, non_blocking=True)
      output, loss = trainer.step(input, target, criterion, minimize=False)
      end = time.time()

      # get some batch statistics
      trainer.log_loss_values({
        'train_{}_batch_{}'.format(part_id, opt.hidden_objective): loss,
      })

      # print statistics
      running_loss += loss  # revert the sign for loss logging
      running_time += end - start
      if batch % opt.print_freq == opt.print_freq - 1:
        message = '[part: %d, epoch: %d/%d, batch: %5d/%d] loss (%s): %.3f, runtime (s): %.3f' % (
          part_id, epoch + 1, total_epoch, batch + 1, total_batch,
          opt.hidden_objective,
          running_loss / opt.print_freq,
          running_time / opt.print_freq
        )
        logger.info(message)

        running_loss = 0.
        running_time = 0.

    # validate
    if epoch % opt.val_freq == opt.val_freq - 1:
      hidden_obj, total = 0, 0
      total_batch_val = len(val_loader)
      for i, (input, target) in enumerate(val_loader):
        logger.info('batch: {}/{}'.format(i + 1, total_batch_val))
        input, target = \
          input.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = trainer.get_eval_output(input)

        hidden_obj += criterion(output, target).item()
        total += 1

      # TODO this is incorrect if batch_size does not divide val set size
      hidden_obj /= total
      trainer.log_loss_values({
        'val_{}_{}'.format(part_id, opt.hidden_objective): hidden_obj
      })

      message = '[part %d, epoch: %d] val %s: %.3f' % (
        part_id, epoch + 1, opt.hidden_objective, hidden_obj)
      logger.info(message)
      if opt.schedule_lr:
        trainer.scheduler_step(hidden_obj)
      trainer.save(epoch, hidden_obj, model_name='net_part{}.pth'.format(part_id))

  logger.info('Part %d training finished!' % part_id)

  # load best part for the next training session
  logger.info('Before proceeding, load saved checkpoint for training the next part...')
  # TODO too hacky
  tmp, opt.checkpoint_dir = opt.checkpoint_dir, opt.save_dir
  trainer.load('net_part{}.pth'.format(part_id))
  opt.checkpoint_dir = tmp
