"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import kernet.utils as utils
from kernet.parsers.base_parser import BaseParser

class TestParser(BaseParser):
  def initialize(self, parser):
    super().initialize(parser)
    parser.add_argument('--adversarial', type=utils.str2bool,
                        nargs='?', const=True, default=False,
      help='If specified, report the test performance under FGM and PGD attacks as well. "True" or "t" will be parsed as True (bool); "False" or "f" as False.')
    parser.add_argument('--adversary_eps', type=float, default=0.3,
      help='Total epsilon for FGM and PGD attacks.')
    parser.add_argument('--adversary_norm', type=str, default='inf', choices=['inf', '2'],
      help='Norm for FGM and PGD attacks.')
    parser.add_argument('--pgd_step_eps', type=float, default=0.01,
      help='Per-step epsilon for PGD attack.')
    parser.add_argument('--pgd_n_steps', type=int, default=50,
      help='Number of attack steps for PGD attack.')

    self.is_train = False
    return parser
