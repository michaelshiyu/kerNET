"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch

import kernet.models as models
import kernet.datasets as datasets
import kernet.utils as utils
from kernet.engines import train
from kernet.parsers import TrainParser
from kernet.trainers.trainer import Trainer


def modify_commandline_options(parser, **kwargs):
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--momentum', type=float, default=.9,
                        help='Momentum for the SGD optimizer.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='L2 regularization on the model weights.')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='The number of training epochs.')
    return parser


def main():
    opt = TrainParser().parse()

    # set up logger
    utils.set_logger(opt=opt, filename='train.log', filemode='w')

    if opt.seed:
        utils.make_deterministic(opt.seed)

    model = models.get_model(opt)
    model = model.to(device)
    if opt.multi_gpu and device == 'cuda':
        model = torch.nn.DataParallel(model)
    if opt.loss == 'xe':
        criterion = torch.nn.CrossEntropyLoss()
    elif opt.loss == 'nll':
        criterion = torch.nn.NLLLoss()
    elif opt.loss == 'hinge':
        criterion = torch.nn.MultiMarginLoss()
    optimizer = utils.get_optimizer(opt, params=model.parameters(), lr=opt.lr,
                                    weight_decay=opt.weight_decay, momentum=opt.momentum)
    trainer = Trainer
    trainer = trainer(opt=opt, model=model, optimizer=optimizer,
                      val_metric_name='acc (%)', val_metric_obj='max')
    if hasattr(model, 'update_centers'):
        trainer.update_centers_eval = lambda: utils.update_centers_eval(model)
    loader, val_loader = datasets.get_dataloaders(opt)

    if opt.load_model:
        trainer.load()

    # save init model
    trainer.save(
        epoch=trainer.start_epoch - 1,
        val_metric_value=trainer.best_val_metric,
        force_save=True
    )

    utils.update_centers_eval(model)
    train(opt, n_epochs=opt.n_epochs, trainer=trainer, loader=loader, val_loader=val_loader,
          criterion=criterion, device=device)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    main()
