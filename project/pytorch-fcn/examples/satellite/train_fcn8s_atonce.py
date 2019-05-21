#!/usr/bin/env python
from torchsummary import summary
import argparse
import datetime
import os
import os.path as osp
import torch
import torch.nn as nn
import yaml
import sys
import lovasz_losses as L
sys.path.insert(0, '../../')
import torchfcn
from train_fcn32s import get_parameters
from train_fcn32s import git_hash
import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil
import fcn
import numpy as np
import pytz
import scipy.misc
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import time
import datetime
from statistics import mean
import skimage
import cv2
import random
from early_stopping import EarlyStoppingIoU
import warnings
from weight_init import weight_init
warnings.filterwarnings("ignore")

# torch.manual_seed(12)
# torch.cuda.manual_seed(12)
# np.random.seed(12)
# random.seed(12)
# torch.backends.cudnn.deterministic=True

#Experiments:
    #Try VGGNet with requires_grad=True
    #Add scheduler
    #Add weight decay and momentum
    #Try Adam optimiser 0.90,0.99 etc
    #Try Batch Size difference
    #Try architecture FCN8sAtOnce, FCN16s, FCN32s etc
    #Try different types of preprocessing of the data
    #Try get different loss function to work

#Things to do:
    #Plot curve of experiment results/write script for future use
    #Make a testing/validation set not just validation
    #Visualise some of the models with good IoU/low file count quick train and Visualise


here = osp.dirname(osp.abspath(__file__))

#batch_size = 1
n_class    = 2

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))


start_time = time.time()
# def get_free_gpu():
#     os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#     memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
#     return np.argmax(memory_available)
#
# gpu_used = int(get_free_gpu())

early_stopping = EarlyStoppingIoU(patience=5)


def train(fcn_model, args, optimizer, criterion, scheduler, train_loader, val_loader, filename=None, filename1=None):
    epochs = args.epochs

    for epoch in range(epochs):
        if args.scheduler == True:
            scheduler.step()

        ts = time.time()

        for iter, (inputs, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            if use_gpu:
                #print("CUDA")
                inputs = Variable(inputs.cuda(3))
                labels = Variable(labels.cuda(3))
            else:
                #print("NO CUDA")
                inputs, labels = Variable(inputs), Variable(labels)

            N, c, h, w = inputs.shape

            outputs = fcn_model(inputs)

            # print("Pred: ", outputs)#.squeeze().shape)
            # print("Labels: ", np.unique(labels.cpu().numpy(), return_counts=True))#[0, :, :, :].shape)

            #outputs = outputs[:,1:,:]
            if args.loss == "CE":
                loss = criterion(outputs, labels)
            else:
                loss = L.lovasz_softmax(outputs, labels, classes=[1])

            loss.backward()

            optimizer.step()

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        print("epoch: {}, loss: {}".format(epoch, loss.data.item()))

        # if args.file == True:
        #     csv_file = open(filename, "a")
        #     csv_file.write(str(epoch) + "," + str(loss.data.item()) + ",")
        #     csv_file.close()


        val(fcn_model, epoch, args, criterion, val_loader, filename, filename1)

def val(fcn_model, epoch, args, criterion, val_loader, filename=None, filename1=None):
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    # pixel_background = []
    # pixel_building = []
    iteration = 0
    val_loss = 0
    count = 0
    for iter, (data, target) in enumerate(val_loader):

        if use_gpu:
            #print("CUDA")
            inputs = Variable(data.cuda(3))
        else:
            #print("NO CUDA")
            inputs = Variable(data)

        output = fcn_model(inputs)

        if args.loss == "CE":
            val_loss += criterion(output, target.cuda(3)).item()
        else:
            val_loss += L.lovasz_softmax(output, target.cuda(3), classes=[1]).item()

        count = count + 1

        output = output.data.cpu().numpy()

        N, c, h, w = output.shape

        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
        #pred = output.transpose(0, 2, 3, 1).reshape(-1, 1).argmax(axis=1).reshape(N, h, w)

        target = target.cpu().numpy().reshape(N, h, w)

        # pixel_building.append(np.unique(target, return_counts=True)[1].item(1))
        # pixel_background.append(np.unique(target, return_counts=True)[1].item(0))

        # N, c, h, w = inputs.shape
        # input_print = inputs.cpu().numpy().reshape(c, h, w).transpose(1,2,0)

        if (iter+1) % 700 == 0:
            now = datetime.datetime.now()
            #if args.file == True:
            img_name  = "predictions/" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "-" + str(epoch) + "-" + str(iter) + "-prediction.jpg"
            targ_name = "predictions/" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "-" + str(epoch) + "-" + str(iter) + "-truth.jpg"
            #inp_name  = "predictions/" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "-" + str(epoch) + "-" + str(iter) + "-raw.jpg"
            scipy.misc.imsave(img_name, pred.squeeze())
            scipy.misc.imsave(targ_name, target.squeeze())
            #scipy.misc.imsave(inp_name, input_print)


        for p, t in zip(pred, target):
            # total_ious.append(L.iou_binary(p, t))
            total_ious.append(L.iou_binary(p, t))
            pixel_accs.append(pixel_acc(p, t))


        iteration += 1

    val_loss /= count
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch: {}, pix_acc: {},  IoU: {}, val_loss: {}".format(epoch, pixel_accs, np.mean(total_ious), val_loss))

    # if args.file == True:
    #     csv_file = open(filename, "a")
    #     csv_file.write(str(pixel_accs) + "," + str(np.mean(total_ious)) + "," + str(val_loss) + "\n")
    #     csv_file.close()

    early_stopping(np.mean(total_ious))#, model)

    if early_stopping.early_stop:
        print("Early stopping")
        end_time = time.time()
        total_time = end_time - start_time
        print(total_time)
        timings_file = open(filename1, "a")
        timings_file.write(str(total_time) + " / " + str(epoch) + "\n")
        # model_name = "./best_fcn_models/" + filename + ".pth"
        # torch.save(fcn_model.state_dict(), model_name)
        sys.exit()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--lr', type=float, default=0.0000001, help='Learning rate',
    )
    parser.add_argument(
        '--scheduler', nargs="?", type=str2bool, default=False, help='Scheduler?',
    )
    parser.add_argument(
        '--optimiser', nargs="?", type=str, default="RMSprop", help='Optimiser used',
    )
    parser.add_argument(
        '--epochs', nargs="?", type=int, default=100, help='The experiment\'s epoch budget'
    )
    # parser.add_argument(
    #     '--step_size', type=int, default=50, help='step size',
    # )
    parser.add_argument(
        '--w_decay', type=float, default=0.00005, help='weight decay',
    )
    # parser.add_argument(
    #     '--gamma', type=float, default=0.5, help='gamma',
    # )
    parser.add_argument(
        '--momentum', type=float, default=0.90, help='momentum',
    )
    parser.add_argument(
        '--file', type=str2bool, default=False, help='Do you want to output data to csv?',
    )
    parser.add_argument(
        '--loss', type=str, default="CE", help='Loss function',
    )
    parser.add_argument(
        '--batch_size', type=int, default=16, help='input batch size for training (default: 16)'
    )
    parser.add_argument(
        '--seed', type=int, default=12, help='random seed (default: 12)'
    )
    parser.add_argument(
        '--w1', type=float, default=1.0, help='weight for CE on background'
    )
    parser.add_argument(
        '--w2', type=float, default=4.5, help='weight for CE on building'
    )
    parser.add_argument(
        '--hf', type=str2bool, default=False, help='Horizontal Flip',
    )
    parser.add_argument(
        '--vf', type=str2bool, default=False, help='Vertical Flip',
    )
    parser.add_argument(
        '--rr', type=str2bool, default=False, help='Random Rotate',
    )
    parser.add_argument(
        '--FCN', type=int, default=8, help='FCNXs'
    )

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

    vgg_model = torchfcn.models.VGGNet(requires_grad=True, remove_fc=True)
    fcn_model = None

    if args.FCN == 2:
        fcn_model = torchfcn.models.FCN2s(pretrained_net=vgg_model, n_class=2)
    elif args.FCN == 4:
        fcn_model = torchfcn.models.FCN4s(pretrained_net=vgg_model, n_class=2)
    elif args.FCN == 8:
        fcn_model = torchfcn.models.FCN8s(pretrained_net=vgg_model, n_class=2)
    elif args.FCN == 16:
        fcn_model = torchfcn.models.FCN16s(pretrained_net=vgg_model, n_class=2)
    elif args.FCN == 32:
        fcn_model = torchfcn.models.FCN32s(pretrained_net=vgg_model, n_class=2)

    fcn_model.apply(weight_init)
    early_stopping = EarlyStoppingIoU(patience=5)
    if use_gpu:
        ts = time.time()
        vgg_model = vgg_model.cuda(3)
        fcn_model = fcn_model.cuda(3)
    #vgg_model = nn.DataParallel(vgg_model)
    #fcn_model = nn.DataParallel(fcn_model)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))



    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SatelliteDataset2(split='train', transform=True, hf=args.hf, vf=args.vf, rr=args.rr),
        batch_size=args.batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SatelliteDataset2(split='val', transform=False),
        batch_size=args.batch_size, shuffle=True)

    if args.optimiser == "RMSprop":
        optimizer = torch.optim.RMSprop(fcn_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    elif args.optimiser == "Adam":
        optimizer = torch.optim.Adam(fcn_model.parameters(), lr=args.lr, betas=(0.9,0.999))

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1,4.5]).cuda(3))

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) # decay LR by a factor of 0.5 every 30 epochs
    scheduler = None

    if args.file == True:
        filename1 = "fcn_types_exp/timings.txt"
        filename =  datetime.datetime.utcnow().strftime("%H:%M:%S")
        filename = "fcn_types_exp/" + filename + ".csv"
        #filename = "experiment_results_fcn_best/" + filename + ".csv"
        # csv_file = open(filename, "a")
        # csv_file.write(str(args) + "\n")
        # csv_file.write("epoch,loss,pixel_acc,IoU,val_loss\n")
        # csv_file.close()

        timings_file = open(filename1, "a")
        timings_file.write(str(args) + "\n")
        timings_file.close()

        train(fcn_model, args, optimizer, criterion, scheduler, train_loader, val_loader, filename, filename1)

        end_time = time.time()
        total_time = end_time - start_time
        timings_file = open(filename1, "a")
        timings_file.write(str(total_time) + "\n")
        # model_name = "./best_fcn_models/" + filename + ".pth"
        # torch.save(fcn_model.state_dict(), model_name)
    else:
        train(fcn_model, args, optimizer, criterion, scheduler, train_loader, val_loader)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

if __name__ == '__main__':
    main()

