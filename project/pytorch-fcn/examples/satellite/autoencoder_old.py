import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torchsummary import summary
import os
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '../../')
import torchfcn
import tqdm
import numpy as np
import random
import time

torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)
torch.backends.cudnn.deterministic=True

if not os.path.exists('./autoencoded_pics'):
    os.mkdir('./autoencoded_pics')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 512, 512)
    return x

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


num_epochs = 100
batch_size = 16
learning_rate = 1e-6

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

filename = "2.csv"

train_loader = torch.utils.data.DataLoader(
    torchfcn.datasets.SatelliteDataset(split='train', transform=True),
    batch_size=batch_size, shuffle=True)

# val_loader = torch.utils.data.DataLoader(
#     torchfcn.datasets.SatelliteDataset(split='val', transform=True),
#     batch_size=batch_size, shuffle=True)

output_file = open("autoencoder_results/" + filename, "w")
output_file.write("epoch,loss\n")
output_file.close()

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

gpu_used = int(get_free_gpu())
model = AutoEncoderConv().cuda(gpu_used)
model = nn.DataParallel(model)
summary(model, input_size=(1, 512, 512))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for i, (_, img) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),desc='Train epoch=%d' % epoch, ncols=80, leave=False):

        img = img.float()
        img = img[:, np.newaxis, :, :]
        img = Variable(img.cuda(gpu_used))
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================

        if (i+1) % len(train_loader) == 0:
            #pic = to_img(output.cpu().data)
            #print("aye la")
            #pic = output[0].detach().cpu().squeeze().numpy()
            #matplotlib.image.imsave('./autoencoded_pics/image_{}.png'.format(epoch), pic, cmap='Greys_r')
            torchvision.utils.save_image(Variable(output).data.cpu(), './autoencoded_pics/real_image_{}.png'.format(epoch))
            torchvision.utils.save_image(Variable(img).data.cpu(), './autoencoded_pics/real_image_{}.png'.format(epoch))

    print('epoch [{}/{}], loss:{}'.format(epoch+1, num_epochs, str(loss.data.item())))
    output_file = open("autoencoder_results/" + filename, "a")
    output_file.write(str(epoch) + "," + str(loss.data.item()) + "\n")
    output_file.close()
