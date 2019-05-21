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
import argparse
import lovasz_losses as L
import datetime
from early_stopping import EarlyStopping
import fcn
import warnings
from weight_init import weight_init
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='CAE')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=75,
                    help='number of epochs to train (default: 50)')
parser.add_argument('--seed', type=int, default=12, metavar='S',
                    help='random seed (default: 12)')
parser.add_argument('--lr', type=float, default=0.0000001,
                    help='Learning rate')
parser.add_argument('--loss', type=str, default="MSE",
                    help='Loss function')
parser.add_argument('--optimiser', nargs="?", type=str, default="RMSprop",
                    help='Optimiser used')
args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic=True

filename =  datetime.datetime.utcnow().strftime("%H:%M:%S")
filename = "experiment_results_autoencoders/" + filename + ".csv"
csv_file = open(filename, "a")
csv_file.write(str(args) + "\n")
csv_file.write("epoch,train_loss,val_loss\n")
csv_file.close()

train_loader = torch.utils.data.DataLoader(
    torchfcn.datasets.SatelliteDataset(split='train', transform=True),
    batch_size=args.batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(
    torchfcn.datasets.SatelliteDataset(split='val', transform=True),
    batch_size=args.batch_size, shuffle=True)


# gpu_used = int(get_free_gpu())
model = torchfcn.models.AutoEncoderConv3().cuda()
model.apply(weight_init)

summary(model, input_size=(1, 256, 256))
model = nn.DataParallel(model)
if args.loss == "MSE":
    criterion = nn.MSELoss()

if args.optimiser == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

early_stopping = EarlyStopping(patience=4)


def train(epoch):
    model.train()
    train_loss = 0
    count = 0
    for i, (_, img) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),desc='Train epoch=%d' % epoch, ncols=80, leave=False):

        img = img.float()
        img = img[:, np.newaxis, :, :]
        img = Variable(img.cuda())
        # ===================forward=====================
        output = model(img)
        if args.loss == "BCE":
            loss = L.binary_xloss(output, img)
        else:
            loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        count += 1
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / count))
    if epoch != 1:
        csv_file = open(filename, "a")
        csv_file.write(str(epoch) + "," + str(train_loss / count) + ",")
        csv_file.close()



def val(epoch):
    model.eval()
    val_loss = 0
    count = 0
    with torch.no_grad():
        for i, (_, img) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader),desc='Val epoch=%d' % epoch, ncols=80, leave=False):
            img = img.float()
            img = img[:, np.newaxis, :, :]
            img = Variable(img.cuda())
            output = model(img)
            if args.loss == "BCE":
                val_loss += L.binary_xloss(output, img).item()
            else:
                val_loss += criterion(output, img).item()
            count += 1
            if (i+1) % len(val_loader) == 0:
                #pic = to_img(output.cpu().data)
                #print("aye la")
                #pic = output[0].detach().cpu().squeeze().numpy()
                #matplotlib.image.imsave('./autoencoded_pics/image_{}.png'.format(epoch), pic, cmap='Greys_r')
                torchvision.utils.save_image(Variable(output).data.cpu(), './autoencoded_pics/predicted_image_{}.png'.format(epoch))
                torchvision.utils.save_image(Variable(img).data.cpu(), './autoencoded_pics/real_image_{}.png'.format(epoch))
        # for i, (data, _) in enumerate(test_loader):
        #     data = data.to(device)
        #     recon_batch, mu, logvar = model(data)
        #     test_loss += loss_function(recon_batch, data, mu, logvar).item()
        #     if i == 0:
        #         n = min(data.size(0), 8)
        #         comparison = torch.cat([data[:n],
        #                               recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
        #         save_image(comparison.cpu(),
        #                  'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    val_loss /= count
    print('====> Validation set loss: {:.4f}'.format(val_loss))
    if epoch != 1:
        csv_file = open(filename, "a")
        csv_file.write(str(val_loss) + "\n")
        csv_file.close()

    early_stopping(val_loss)#, model)

    if early_stopping.early_stop:
        print("Early stopping")
        torch.save(model.state_dict(), "autoencoder_4lay_2.pth")
        # torch.save(model, "autoencoder1.pth")
        sys.exit()

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        val(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')
    torch.save(model.state_dict(), "autoencoder_4lay_2.pth")
    #
    # # #Later to restore:
    # # model.load_state_dict(torch.load(filepath))
    # # model.eval()
    #
    # #or
    #
    # torch.save(model, "autoencoder1.pth")

    # Then later:
    #model = torch.load(filepath)

