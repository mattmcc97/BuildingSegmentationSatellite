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
from torchvision import models
from weight_init import weight_init
warnings.filterwarnings("ignore")
import matplotlib

torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)
torch.backends.cudnn.deterministic=True


here = osp.dirname(osp.abspath(__file__))

n_class    = 2

use_gpu = torch.cuda.is_available()

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

gpu_used = int(get_free_gpu())


vgg_model = torchfcn.models.VGGNet(requires_grad=True, remove_fc=True)
fcn_model = torchfcn.models.FCN8s(pretrained_net=vgg_model, n_class=2)
fcn_model.apply(weight_init)

model2 = torchfcn.models.AutoEncoderConv3()
model2.load_state_dict(torch.load("trained_models/autoencoder_4lay_2.pth"), strict=False)




#FREEZE DECODER
for p in model2.parameters():
   p.requires_grad = False

early_stopping = EarlyStoppingIoU(patience=7)

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda(gpu_used)
    fcn_model = fcn_model.cuda(gpu_used)
    model2 = model2.cuda(gpu_used)
    transform_conv = torchfcn.models.TransformConv().cuda(gpu_used)
    # fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

# vgg_model = nn.DataParallel(vgg_model)
# fcn_model = nn.DataParallel(fcn_model)
# model2 = nn.DataParallel(model2)

def train(args, optimizer, criterion, criterion2, scheduler, train_loader, val_loader, test_loader, filename=None):
    epochs = args.epochs

    for epoch in range(epochs):
        if args.scheduler == True:
            scheduler.step()

        ts = time.time()

        for iter, (inputs, labels) in tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader),
                desc='Train epoch=%d' % epoch, ncols=80, leave=False):

            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(inputs.cuda(gpu_used))
                labels = Variable(labels.cuda(gpu_used))
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            N, c, h, w = inputs.shape

            intermediate_input_from_fcn = fcn_model.pretrained_net(inputs)['x4']
            intermediate_input_from_fcn = transform_conv(intermediate_input_from_fcn)

            outputs = fcn_model(inputs)
            N, c, h, w = outputs.shape

            #EXTRACT INTERMEDIATE INPUT
            outputs2 = model2.decode(intermediate_input_from_fcn, iter)

            if args.loss == "CE":
                loss1 = criterion(outputs, labels)
                loss2 = criterion2(outputs2, labels.float())
                # print("main loss: ", loss1)
                # print("aux loss: ", loss2)
                #print(loss2)
                loss = loss1 + (args.loss2weight * loss2)
            else:
                loss1 = L.lovasz_softmax(outputs, labels, classes=[1])
                loss2 = criterion2(outputs2, labels.float())
                loss = loss1 + (args.loss2weight * loss2)

            if (iter+1) % 2000 == 0:
                now = datetime.datetime.now()
                #if args.file == True:
                output1_name  = "predictions-combined/" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "-" + str(epoch) + "-" + str(iter) + "-FCN.jpg"
                output2_name  = "predictions-combined/" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "-" + str(epoch) + "-" + str(iter) + "-DECODER.jpg"
                targ_name = "predictions-combined/" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "-" + str(epoch) + "-" + str(iter) + "-truth.jpg"
                inp_name  = "predictions-combined/" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "-" + str(epoch) + "-" + str(iter) + "-raw.jpg"

                outputs = outputs.data.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
                outputs2 = outputs2.data.cpu().numpy()[:, 0, :, :]
                #print(np.unique(outputs2, return_counts=True))
                scipy.misc.imsave(output1_name, outputs.squeeze())
                matplotlib.image.imsave(output2_name, outputs2.squeeze(), cmap='Greys_r')
                #scipy.misc.imsave(output2_name, outputs2.squeeze())
                scipy.misc.imsave(targ_name, labels.cpu().detach().numpy().squeeze())
                scipy.misc.imsave(inp_name, inputs.cpu().detach().numpy().squeeze().transpose(1, 2, 0))

            #a = list(model2.parameters())[0].clone()

            loss.backward()

            optimizer.step()

            #b = list(model2.parameters())[0].clone()
            #CONFIRMATION THAT DECODER IS FROZEN
            #print("Equal?: ", torch.equal(a.data, b.data))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        print("epoch: {}, loss: {}".format(epoch, loss.data.item()))

        if args.file == True:
            csv_file = open(filename, "a")
            csv_file.write(str(epoch) + "," + str(loss.data.item()) + ",")
            csv_file.close()


        val(epoch, args, criterion, val_loader, test_loader, filename)

def val(epoch, args, criterion, val_loader, test_loader, filename=None):
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    iteration = 0
    val_loss = 0
    count = 0
    for iter, (data, target) in tqdm.tqdm(
            enumerate(val_loader), total=len(val_loader),
            desc='Valid iteration=%d' % iteration, ncols=80,
            leave=False):

        if use_gpu:
            inputs = Variable(data.cuda(gpu_used))
        else:
            inputs = Variable(data)

        output = fcn_model(inputs)

        if args.loss == "CE":
            val_loss += criterion(output, target.cuda(gpu_used)).item()
        else:
            val_loss += L.lovasz_softmax(output, target.cuda(gpu_used), classes=[1]).item()

        count = count + 1

        output = output.data.cpu().numpy()

        N, c, h, w = output.shape

        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = target.cpu().numpy().reshape(N, h, w)


        for p, t in zip(pred, target):

            total_ious.append(L.iou_binary(p, t))
            pixel_accs.append(pixel_acc(p, t))


        iteration += 1

    val_loss /= count
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch: {}, pix_acc: {},  IoU: {}, val_loss: {}".format(epoch, pixel_accs, np.mean(total_ious), val_loss))

    if args.file == True:
        csv_file = open(filename, "a")
        csv_file.write(str(pixel_accs) + "," + str(np.mean(total_ious)) + "," + str(val_loss) + "\n")
        csv_file.close()

    early_stopping(np.mean(total_ious))#, model)

    if early_stopping.early_stop:
        print("Early stopping")
        #test_set(test_loader)
        test_set(test_loader, filename)
        sys.exit()


def test_set(test_loader, filename):
    print("TEST SET EVALUATION")
    fcn_model.eval()

    total_ious = []
    pixel_accs = []

    for iter, (data, target) in tqdm.tqdm(
            enumerate(test_loader), total=len(test_loader),
            desc='Test iteration', ncols=80,
            leave=False):

        if use_gpu:
            inputs = Variable(data.cuda())
        else:
            inputs = Variable(data)

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        N, c, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
        target = target.cpu().numpy().reshape(N, h, w)

        if iter % 100 == 0:
            now = datetime.datetime.now()
            #if args.file == True:
            img_name  = "predictions/" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "-" + str(iter) + "-prediction.jpg"
            targ_name = "predictions/" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "-" + str(iter) + "-truth.jpg"
            #inp_name  = "predictions/" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "-" + str(iter) + "-raw.jpg"
            scipy.misc.imsave(img_name, pred.squeeze())
            scipy.misc.imsave(targ_name, target.squeeze())
            #scipy.misc.imsave(inp_name, input_print)

        for p, t in zip(pred, target):
            total_ious.append(L.iou_binary(p, t))
            pixel_accs.append(pixel_acc(p, t))

    pixel_accs = np.array(pixel_accs).mean()
    print("pix_acc: {},  IoU: {}, file: {}".format(pixel_accs, np.mean(total_ious), filename))
    csv_file = open("test_fcn_combined_results" + ".csv", "a")
    csv_file.write(str(np.mean(total_ious)) + "\n")
    csv_file.close()

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
    parser.add_argument(
        '--w_decay', type=float, default=0.00005, help='weight decay',
    )
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
        '--loss2weight', type=float, default=1.0, help='By what factor do you want to weight the auxiliary loss'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16, help='input batch size for training (default: 16)'
    )

    args = parser.parse_args()
    print(args)

    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SatelliteDataset(split='train', transform=True),
        batch_size=args.batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SatelliteDataset(split='val', transform=True),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SatelliteDataset(split='test', transform=True),
        batch_size=1, shuffle=True)

    if args.optimiser == "RMSprop":
        optimizer = torch.optim.RMSprop(fcn_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    elif args.optimiser == "Adam":
        optimizer = torch.optim.Adam(fcn_model.parameters(), lr=args.lr, betas=(0.9,0.999))

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1,4.5]).cuda(gpu_used))
    criterion2 = nn.MSELoss()
    scheduler = None

    if args.file == True:
        filename =  datetime.datetime.utcnow().strftime("%H:%M:%S")
        filename = "experiment_results_fcn_combined_4/" + filename + ".csv"
        csv_file = open(filename, "a")
        csv_file.write(str(args) + "\n")
        csv_file.write("epoch,loss,pixel_acc,IoU,val_loss\n")
        csv_file.close()
        csv_file = open("test_fcn_combined_results" + ".csv", "a")
        csv_file.write(str(args) + "\n")
        #csv_file.write("epoch,loss,pixel_acc,IoU,val_loss\n")
        csv_file.close()
        #train(args, optimizer, criterion, criterion2, scheduler, train_loader, val_loader, filename)
        train(args, optimizer, criterion, criterion2, scheduler, train_loader, val_loader, test_loader, filename)
        test_set(test_loader, filename)
    else:
        train(args, optimizer, criterion, criterion2, scheduler, train_loader, val_loader, test_loader)

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

