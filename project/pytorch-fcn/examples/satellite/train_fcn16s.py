#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

import torch
import yaml
import sys
sys.path.insert(0, '../../')
import torchfcn

from train_fcn32s import get_parameters
from train_fcn32s import git_hash


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--resume', help='checkpoint path')
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument(
        '--max-iteration', type=int, default=100000, help='max iteration'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-12, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
        '--pretrained-model',
        help='pretrained model of FCN32s',
    )
    args = parser.parse_args()

    args.model = 'FCN16s'
    args.git_hash = git_hash()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    #root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SatelliteDataset(split='train', transform=True),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SatelliteDataset(
            split='val', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    model = torchfcn.models.FCN16s(n_class=2)
    start_epoch = 0
    start_iteration = 0
    # if args.resume:
    #     checkpoint = torch.load(args.resume)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     start_iteration = checkpoint['iteration']
    # else:
    #     fcn32s = torchfcn.models.FCN32s()
    #     state_dict = torch.load(args.pretrained_model)
    #     try:
    #         fcn32s.load_state_dict(state_dict)
    #     except RuntimeError:
    #         fcn32s.load_state_dict(state_dict['model_state_dict'])
    #     model.copy_params_from_fcn32s(fcn32s)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': args.lr * 2, 'weight_decay': 0},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_iter=args.max_iteration,
        interval_validate=4000,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
