"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch

import kernet.utils as utils
import kernet.models as models
import kernet.datasets as datasets
from kernet.parsers import TestParser


def test(opt, net, loader):
    correct = 0
    if opt.adversarial:
        correct_fgm, correct_pgd = 0, 0
    total = 0
    net.eval()
    logger.info('Starting testing...')

    with torch.set_grad_enabled(opt.adversarial):
        # adversarial attacks need grad computations
        for i, (input, target) in enumerate(loader):
            logger.info('batch: {}/{}'.format(i, len(loader)))
            input, target = \
                input.to(device, non_blocking=True), target.to(
                    device, non_blocking=True)

            output = net(input, update_centers=False)

            # save tensors for visualization
            raw_data = input if i == 0 else torch.cat((raw_data, input))
            labels = target if i == 0 else torch.cat((labels, target))
            activations = net_head(input)
            all_activations = activations if i == 0 else torch.cat(
                (all_activations, activations))

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if opt.adversarial:
                def net_fn(input): return net(input, update_centers=False)
                input_fgm = fast_gradient_method(
                    net_fn, input, opt.adversary_eps, opt.adversary_norm)
                input_pgd = projected_gradient_descent(net_fn, input, opt.adversary_eps,
                                                       opt.pgd_step_eps, opt.pgd_n_steps, opt.adversary_norm)
                output_fgm = net(input_fgm, update_centers=False)
                output_pgd = net(input_pgd, update_centers=False)
                _, predicted_fgm = torch.max(output_fgm.data, 1)
                _, predicted_pgd = torch.max(output_pgd.data, 1)
                correct_fgm += (predicted_fgm == target).sum().item()
                correct_pgd += (predicted_pgd == target).sum().item()

    logger.info('Accuracy (%): {:.3f}'.format(100 * correct / total))

    if opt.adversarial:
        logger.info('Accuracy under FGM (%): {:.3f}'.format(
            100 * correct_fgm / total))
        logger.info('Accuracy under PGD (%): {:.3f}'.format(
            100 * correct_pgd / total))

    logger.info('Testing finished!')

    return raw_data.cpu().numpy(), labels.cpu().numpy(), all_activations.cpu().numpy()


def visualize(tensor, labels, fig_title, save_name, pca_first=False):
    tensor = tensor.reshape(len(tensor), -1)
    tsne = TSNE(n_components=2, verbose=True)
    if pca_first:
        pca = PCA(n_components=50)
        pca_results = pca.fit_transform(tensor)
        tsne_results = tsne.fit_transform(pca_results)
    else:
        tsne_results = tsne.fit_transform(tensor)

    df = pd.DataFrame(tensor)
    df['y'] = labels.astype(np.int)
    df['labels'] = df['y'].apply(lambda i: str(i))
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure()
    sns.set(font_scale=1.2)
    sns_plot = sns.scatterplot(
        x='tsne-2d-one', y='tsne-2d-two',
        hue='y',
        palette=sns.color_palette('hls', 10),
        data=df,
        legend=False,
        alpha=.3,
    )
    sns_plot.set_title(fig_title)
    fig = sns_plot.get_figure()
    fig.savefig(save_name, bbox_inches='tight')


if __name__ == '__main__':
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
    net.load_state_dict(torch.load(os.path.join(
        opt.checkpoint_dir, 'net.pth'))['state_dict'])

    if hasattr(net, 'update_centers'):
        utils.update_centers_eval(net)

    net_head = torch.nn.Sequential(*list(net.children())[:-1])

    loader = datasets.get_dataloaders(opt)
    raw_data, labels, activations = test(opt, net, loader)

    idx = np.random.permutation(10000)
    visualize(raw_data[idx], labels[idx], '2D t-SNE for Raw Data (After PCA-50)',
              os.path.join(opt.save_dir, 'raw_data.pdf'), pca_first=True)
    visualize(activations[idx], labels[idx], '2D t-SNE for Penultimate Layer Activations',
              os.path.join(opt.save_dir, 'activations.pdf'))
