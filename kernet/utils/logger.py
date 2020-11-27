"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import logging


def set_logger(opt, filename, filemode):
    numeric_level = getattr(logging, opt.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % opt.loglevel)

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )

    # set file handler
    fh = logging.FileHandler(os.path.join(
        opt.save_dir, filename), mode=filemode)
    fh.setLevel(logging.NOTSET)  # log everything to file
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # set console handler
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)  # lower-level logs will not be printed
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if getattr(opt, 'tf_log', None):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            log_dir=os.path.join(opt.save_dir, 'tf_log')
        )
        # give handles on tensorboard methods to this logger
        logger.add_scalar = writer.add_scalar
