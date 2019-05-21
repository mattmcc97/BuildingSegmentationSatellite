# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import matplotlib


class FCN32s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN16s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        # for key, value in output.items():
        #     print(key)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        #print(x5.shape)
        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        #score1 = self.relu(self.deconv1(x5))

        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        #score2 = self.bn1(score + x4)

        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        #score3 = self.bn2(score + x3)

        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        #score4 = self.bn3(self.relu(self.deconv3(score)))

        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        #score5 = self.bn4(self.relu(self.deconv4(score)))

        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        #print(score.shape)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        #score = torch.cat([score1, score2, score3, score4, score5], dim=1) #HYPERCOLUMNS
        return score  # size=(N, n_class, x.H/1, x.W/1)

class TransformConv(nn.Module):
    def __init__(self):
        super(TransformConv, self).__init__()
        self.conv1 = nn.Conv2d(512, 128, 1)
    def forward(self, x):
       return self.conv1(x)

class FCN8sIntermediate1(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net

    def forward(self, x):
        score = self.pretrained_net(x)
        return score

# class FCN8sIntermediate2(nn.Module):
#
#     def __init__(self, pretrained_net, n_class):
#         super().__init__()
#         self.n_class = n_class
#         self.pretrained_net = pretrained_net
#         self.relu    = nn.ReLU(inplace=True)
#         self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn1     = nn.BatchNorm2d(512)
#         self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn2     = nn.BatchNorm2d(256)
#         self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn3     = nn.BatchNorm2d(128)
#         self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn4     = nn.BatchNorm2d(64)
#         self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn5     = nn.BatchNorm2d(32)
#         self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
#
#     def forward(self, x):
#         x5 = x['x5']
#         x4 = x['x4']
#         x3 = x['x3']
#
#         score = self.relu(self.deconv1(x5))
#
#         score = self.bn1(score + x4)
#
#         score = self.relu(self.deconv2(score))
#         score = self.bn2(score + x3)
#
#         score = self.bn3(self.relu(self.deconv3(score)))
#
#         score = self.bn4(self.relu(self.deconv4(score)))
#
#         score = self.bn5(self.relu(self.deconv5(score)))
#         score = self.classifier(score)
#
#         return score

class FCN4s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        #x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x3)
        score = self.relu(self.deconv3(score))
        score = self.bn3(score + x2)
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN2s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x3)
        score = self.relu(self.deconv3(score))
        score = self.bn3(score + x2)
        score = self.relu(self.deconv4(score))
        score = self.bn4(score + x1)
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score  # size=(N, n_class, x.H/1, x.W/1)

# class FCN8sCombined(nn.Module):
#
#     def __init__(self, pretrained_net, n_class):
#         super().__init__()
#         self.n_class = n_class
#         self.pretrained_net = pretrained_net
#         self.relu    = nn.ReLU(inplace=True)
#         self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn1     = nn.BatchNorm2d(512)
#         self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn2     = nn.BatchNorm2d(256)
#         self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn3     = nn.BatchNorm2d(128)
#         self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn4     = nn.BatchNorm2d(64)
#         self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn5     = nn.BatchNorm2d(32)
#         self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
#         #self.autoencoder = AutoEncoderConv().cuda()
#
#     def forward(self, x):
#         output = self.pretrained_net(x)
#         x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
#         x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
#         x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
#
#         score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
#         score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
#         score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
#         score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
#         score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
#         score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
#         score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
#         score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
#
#         return score  # size=(N, n_class, x.H/1, x.W/1)



class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# class AutoEncoderConvAux(nn.Module):
#     def __init__(self, pretrained_net):
#         super(AutoEncoderConvAux, self).__init__()
#         self.pretrained_net = pretrained_net
#
#         self.decoder = nn.Sequential(
#
#             Interpolate(mode='bilinear', scale_factor=2),
#             nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
#
#             Interpolate(mode='bilinear', scale_factor=2),
#             nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
#
#             Interpolate(mode='bilinear', scale_factor=2),
#             nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
#
#             Interpolate(mode='bilinear', scale_factor=2),
#             nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
#
#             Interpolate(mode='bilinear', scale_factor=2),
#             nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
#             nn.ReLU(True),
#
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         output = self.pretrained_net(x)
#         x = output['x5']
#         x = self.decoder(x)
#
#         return x


class Interpolate(nn.Module):
    def __init__(self, mode, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(x, mode=self.mode, scale_factor=self.scale_factor, align_corners=False)
        return x

class AutoEncoderConv(nn.Module):
    def __init__(self):
        super(AutoEncoderConv, self).__init__()
        self.conv2d1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2d = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(2)
        self.interp = Interpolate(mode='bilinear', scale_factor=2)
        self.deconv2d = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1)
        self.deconv2d1 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            #nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            #nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            #nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            #nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            #nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(

            #TEST ADD SOME STRIDE INSTEAD OF UPSAMPLING

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
            #nn.ReLU(True),

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
            #nn.ReLU(True),

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
            #nn.ReLU(True),

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
            #nn.ReLU(True),

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.ReLU(True),
            #Interpolate(mode='bilinear', scale_factor=2),

            #nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        # print()
        # print("START: ", x.shape)
        # print("Start Encode: ", x.shape)
        x = self.encoder(x)
        # print("Finished Encode: ", x.shape)
        #matplotlib.image.imsave('./autoencoded_pics/image.png', x.detach().cpu().squeeze().numpy()[4, :, :], cmap='Greys_r')
        x = self.decoder(x)
        # print("Finished Decode: ", x.shape)

        # n = 0
        # # matplotlib.image.imsave('./autoencoded_pics/image_{}.png'.format(n), x.detach().cpu().squeeze().numpy(), cmap='Greys_r')
        # # n= n + 1
        # #ENCODE
        # x = self.conv2d1(x)
        # x = self.relu(x)
        # x = self.max_pool(x)
        #
        # # y = self.relu(self.deconv2d1(self.interp(x)))
        # # matplotlib.image.imsave('./autoencoded_pics/image_{}.png'.format(n), x.detach().cpu().squeeze().numpy()[4, :, :], cmap='Greys_r')
        # # n= n + 1
        # # print("E1: ", x.shape)
        # x = self.relu(self.conv2d(x))
        # x = self.max_pool(x)
        #
        # # matplotlib.image.imsave('./autoencoded_pics/image_{}.png'.format(n), x.detach().cpu().squeeze().numpy()[4, :, :], cmap='Greys_r')
        # # n= n + 1
        # # print("E2: ", x.shape)
        # x = self.relu(self.conv2d(x))
        # x = self.max_pool(x)
        # #
        # # matplotlib.image.imsave('./autoencoded_pics/image_{}.png'.format(n), x.detach().cpu().squeeze().numpy()[4, :, :], cmap='Greys_r')
        # # n= n + 1
        # # print("E3: ", x.shape)
        # x = self.relu(self.conv2d(x))
        # x = self.max_pool(x)
        #
        # matplotlib.image.imsave('./autoencoded_pics/image_{}.png'.format(n), x.detach().cpu().squeeze().numpy()[4, :, :], cmap='Greys_r')
        # n= n + 1
        # # print("E4: ", x.shape)
        # # x = self.relu(self.conv2d(x))
        # # x = self.max_pool(x)
        #
        # # matplotlib.image.imsave('./autoencoded_pics/image_{}.png'.format(n), x.detach().cpu().squeeze().numpy()[4, :, :], cmap='Greys_r')
        # # n= n + 1
        # # print("E5: ", x.shape)
        #
        # # #DECODE
        # # x = self.relu(self.deconv2d(self.interp(x)))
        # # matplotlib.image.imsave('./autoencoded_pics/image_{}.png'.format(n), x.detach().cpu().squeeze().numpy()[4, :, :], cmap='Greys_r')
        # # n= n + 1
        # # print("D1: ", x.shape)
        # x = self.relu(self.deconv2d(self.interp(x)))
        # matplotlib.image.imsave('./autoencoded_pics/image_{}.png'.format(n), x.detach().cpu().squeeze().numpy()[4, :, :], cmap='Greys_r')
        # n= n + 1
        # # print("D2: ", x.shape)
        # x = self.relu(self.deconv2d(self.interp(x)))
        # # matplotlib.image.imsave('./autoencoded_pics/image_{}.png'.format(n), x.detach().cpu().squeeze().numpy()[4, :, :], cmap='Greys_r')
        # # n= n + 1
        # # print("D3: ", x.shape)
        # x = self.relu(self.deconv2d(self.interp(x)))
        # # matplotlib.image.imsave('./autoencoded_pics/image_{}.png'.format(n), x.detach().cpu().squeeze().numpy()[4, :, :], cmap='Greys_r')
        # # n= n + 1
        # # print("D4: ", x.shape)
        # x = self.relu(self.deconv2d1(self.interp(x)))
        # # matplotlib.image.imsave('./autoencoded_pics/image_{}.png'.format(n), x.detach().cpu().squeeze().numpy(), cmap='Greys_r')
        # # n= n + 1
        # # print("D5: ", x.shape)
        #
        # self.tanh(x)

        return x

class AutoEncoderConv2(nn.Module):
    def __init__(self):
        super(AutoEncoderConv2, self).__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.Tanh()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, y, iter):
        # if iter == 2300:
            # matplotlib.image.imsave('./autoencoded_pics/intermediate.png', y.detach().cpu().squeeze().numpy()[0, :, :], cmap='Greys_r')
        y = self.decoder(y)
        #print(y.shape)
        # if iter == 2300:
            # matplotlib.image.imsave('./autoencoded_pics/decoded.png', y.detach().cpu().squeeze().numpy()[:, :], cmap='Greys_r')/
        return y

    def forward(self, x):
        mid_x = self.encoder(x)
        final_x = self.decoder(mid_x)

        return final_x

class AutoEncoderConv3(nn.Module):
    def __init__(self):
        super(AutoEncoderConv3, self).__init__()
        self.conv2d1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2d = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(2)
        self.interp = Interpolate(mode='bilinear', scale_factor=2)
        self.deconv2d = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1)
        self.deconv2d1 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),

            # nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),

            # Interpolate(mode='bilinear', scale_factor=2),
            # nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(128, 32, kernel_size=3, padding=1),

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.Tanh()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, y, iter):
        # if iter == 2300:
            # matplotlib.image.imsave('./autoencoded_pics/intermediate.png', y.detach().cpu().squeeze().numpy()[0, :, :], cmap='Greys_r')
        y = self.decoder(y)
        #print(y.shape)
        # if iter == 2300:
            # matplotlib.image.imsave('./autoencoded_pics/decoded.png', y.detach().cpu().squeeze().numpy()[:, :], cmap='Greys_r')/
        return y

    def forward(self, x):
        mid_x = self.encoder(x)
        final_x = self.decoder(mid_x)

        return final_x



if __name__ == "__main__":
    batch_size, n_class, h, w = 10, 20, 160, 160

    # test output size
    vgg_model = VGGNet(requires_grad=True)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, 224, 224))
    output = vgg_model(input)
    assert output['x5'].size() == torch.Size([batch_size, 512, 7, 7])

    fcn_model = FCN32s(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCN16s(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    print("Pass size check")

    # test a random batch, loss should decrease
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    y = torch.autograd.Variable(torch.randn(batch_size, n_class, h, w), requires_grad=False)
    for iter in range(10):
        optimizer.zero_grad()
        output = fcn_model(input)
        output = nn.functional.sigmoid(output)
        loss = criterion(output, y)
        loss.backward()
        print("iter{}, loss {}".format(iter, loss.data[0]))
        optimizer.step()

