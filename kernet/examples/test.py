"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import json
import logging

import torch

import kernet_future.utils as utils
import kernet_future.models as models
import kernet_future.datasets as datasets
from kernet_future.parsers import TestParser


def test(opt, net, loader):
  correct = 0
  if opt.adversarial:
    correct_fgm, correct_pgd = 0, 0
  total = 0
  net.eval()
  logger.info('Starting testing...')

  with torch.set_grad_enabled(opt.adversarial):
    # adversarial attacks need grad computations
    n_batches = len(loader)
    for i, (input, target) in enumerate(loader):
      logger.info('batch: {}/{}'.format(i + 1, n_batches))
      input, target = \
        input.to(device, non_blocking=True), target.to(device, non_blocking=True)

      output = net(input, update_centers=False)
      _, predicted = torch.max(output.data, 1)
      total += target.size(0)
      correct += (predicted == target).sum().item()
      
      if opt.adversarial:
        net_fn = lambda input: net(input, update_centers=False)
        input_fgm = fast_gradient_method(net_fn, input, opt.adversary_eps, opt.adversary_norm)
        input_pgd = projected_gradient_descent(net_fn, input, opt.adversary_eps,
            opt.pgd_step_eps, opt.pgd_n_steps, opt.adversary_norm)
        output_fgm = net(input_fgm, update_centers=False)
        output_pgd = net(input_pgd, update_centers=False)
        _, predicted_fgm = torch.max(output_fgm.data, 1)
        _, predicted_pgd = torch.max(output_pgd.data, 1)
        correct_fgm += (predicted_fgm == target).sum().item()
        correct_pgd += (predicted_pgd == target).sum().item()

  acc = 100 * correct / total
  results = {'accuracy (%)': acc}
  logger.info('Accuracy (%): {:.3f}'.format(acc))

  if opt.adversarial:
    acc_fgm = 100 * correct_fgm / total
    logger.info('Accuracy under FGM (%): {:.3f}'.format(acc_fgm))
    acc_pgd = 100 * correct_pgd / total
    logger.info('Accuracy under PGD (%): {:.3f}'.format(acc_pgd))
    results['accuracy under FGM (%)'] = acc_fgm
    results['accuracy under PGD (%)'] = acc_pgd

  with open(os.path.join(opt.save_dir, 'test.json'), 'w') as out:
    json.dump(results, out, indent=2)
  logger.info('Testing finished!')


if __name__=='__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  opt = TestParser().parse()

  # set up logger
  utils.set_logger(opt=opt, filename='test.log', filemode='a')
  logger = logging.getLogger()

  if opt.adversarial:
    from cleverhans.future.torch.attacks import \
        fast_gradient_method, projected_gradient_descent

  net = models.get_model(opt) 
  net = net.to(device)
  if opt.multi_gpu and device == 'cuda':
    net = torch.nn.DataParallel(net)
  net.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, 'net.pth'))['state_dict'])

  if hasattr(net, 'update_centers'):
    utils.update_centers_eval(net)

  loader = datasets.get_dataloaders(opt)
  test(opt, net, loader)
