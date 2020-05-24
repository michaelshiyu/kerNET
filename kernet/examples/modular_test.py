"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Test a trained model as several modules. The hidden modules will be tested using
the specified proxy hidden objective. The underlying model must support the split
method.
"""
import os
import json
import logging
import collections

import torch

import kernet.utils as utils
import kernet.models as models
import kernet.layers.loss as losses
import kernet.datasets as datasets
from kernet.parsers import TestParser


loss_names = ['srs_raw', 'srs_nmse', 'srs_alignment', 'srs_upper_tri_alignment', 'srs_contrastive', 'srs_log_contrastive']


# TODO duplicate argument additions
# TODO support adversarial attacks (need attackers that take arbitrary loss functions for the hidden proxies)
def modify_commandline_options(parser, **kwargs):
  parser.add_argument('--hidden_objective',
                      choices=loss_names + [_ + '_neo' for _ in loss_names],
                      default='srs_alignment',
                      help='Proxy hidden objective.')
  parser.add_argument('--use_proj_head', type=utils.str2bool,
                      nargs='?', const=True, default=False,
                      help='Whether to attach a trainable two-layer MLP projection head to the ' + \
                           'output of the hidden modules during training. If added, the heads project ' + \
                      'all activations to the same Euclidean space with dimension determined by head_size.')
  parser.add_argument('--split_mode', type=int, default=1,
                      help='The mode to perform the split. Effective only for certain networks.')
  parser.add_argument('--head_size', type=int, default=512,
                      help='Output size of the projection head.')
  parser.add_argument('--n_classes', type=int, default=10,
                      help='The number of classes in the data.')
  return parser


def modular_test(opt, model, loader):
  modules, _ = model.split(n_parts=opt.n_parts, mode=opt.split_mode)
  output_layer = list(model.children())[-1]
  hidden_criterion = getattr(losses, opt.hidden_objective)(output_layer.phi, opt.n_classes)
  correct = 0
  total = 0
  results = collections.defaultdict(int)

  net.eval()
  n_batches = len(loader)
  logger.info('Starting testing...')
  with torch.no_grad():
    for i, (input, target) in enumerate(loader):
      logger.info('batch: {}/{}'.format(i, len(loader)))
      input, target = \
        input.to(device, non_blocking=True), target.to(device, non_blocking=True)

      for j, m in enumerate(modules[:-1]):
        output = m(input)
        results[f'hidden loss {j + 1}'] += (hidden_criterion(output, target).item() / n_batches)

      output = model(input, update_centers=False)
      _, predicted = torch.max(output.data, 1)
      total += target.size(0)
      correct += (predicted == target).sum().item()

  acc = 100 * correct / total
  results['accuracy (%)'] = acc

  with open(os.path.join(opt.save_dir, 'modular_test.json'), 'w') as out:
    json.dump(results, out, indent=2)

  logger.info('\n'.join(f'{k}: {v:.3f}' for k, v in results.items()))
  logger.info('Testing finished!')


if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  opt = TestParser().parse()

  # set up logger
  utils.set_logger(opt=opt, filename='modular_test.log', filemode='a')
  logger = logging.getLogger()

  net = models.get_model(opt)
  net = net.to(device)
  if opt.multi_gpu and device == 'cuda':
    net = torch.nn.DataParallel(net)
  net.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, 'net.pth'))['state_dict'])

  if hasattr(net, 'update_centers'):
    utils.update_centers_eval(net)

  loader = datasets.get_dataloaders(opt)
  modular_test(opt, net, loader)
