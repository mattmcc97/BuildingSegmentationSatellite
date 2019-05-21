import collections
import os.path as osp
import os
import numpy as np
import PIL.Image
import scipy.io
import torch
import scipy.misc
from torch.utils import data
import random
import skimage

#root_img_dir = '../../data1/s1515679/Train_Data/'
num_class = 2
# means = np.array([103.939, 116.779, 123.68])  # mean of three channels in the order of BGR
# h, w      = 650, 650
# new_h, new_w = 640, 640
class_names = np.array([
    'building',
    'not_building',
])

dim = 256

# image_transforms = {
#     # Train uses data augmentation
#     'train':
#     transforms.Compose([
#         transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
#         transforms.RandomRotation(degrees=15),
#         transforms.ColorJitter(),
#         transforms.RandomHorizontalFlip(),
#         transforms.CenterCrop(size=224),  # Image net standards
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])  # Imagenet standards
#     ]),
#     # Validation does not use augmentation
#     'valid':
#     transforms.Compose([
#         transforms.Resize(size=256),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

class SatelliteDataset(data.Dataset):

    class_names = np.array([
        'building',
        'not_building',
    ])


    def __init__(self, split, transform=False):
        #self.n_class   = n_class
        self._transform = transform
        self.split = split
        self.n_class = num_class
        #self.means = means

        dataset_dir = osp.join('', '/data1/s1515679/')
        self.files = collections.defaultdict(list)
        if split == 'train':
            train_dataset = osp.join(dataset_dir, 'Train_Data')
            files = 0
            for filename in os.listdir(train_dataset):
                if filename.endswith(".tif"):# and files < 101:
                    img_file = osp.join(train_dataset, filename)
                    lbl_file = osp.join(dataset_dir, 'Mask-Data/RGB-PanSharpen-Masks/' + filename)
                    self.files[split].append({
                        'img': img_file,
                        'lbl': lbl_file,
                    })
                    files +=1
        elif split == 'val':
            val_dataset = osp.join(dataset_dir, 'Val_Data')
            files = 0
            for filename in os.listdir(val_dataset):
                if filename.endswith(".tif"):# and files < 15:
                    img_file = osp.join(val_dataset, filename)
                    lbl_file = osp.join(dataset_dir, 'Mask-Data/RGB-PanSharpen-Masks/' + filename)
                    self.files[split].append({
                        'img': img_file,
                        'lbl': lbl_file,
                    })
                    files +=1
        elif split == 'test':
            val_dataset = osp.join(dataset_dir, 'Test_Data')
            files = 0
            for filename in os.listdir(val_dataset):
                if filename.endswith(".tif"):# and files < 15:
                    img_file = osp.join(val_dataset, filename)
                    lbl_file = osp.join(dataset_dir, 'Mask-Data/RGB-PanSharpen-Masks/' + filename)
                    self.files[split].append({
                        'img': img_file,
                        'lbl': lbl_file,
                    })
                    files +=1
        #print(split)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        #print()
        #print("IMAGE: ", img_file)
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        img = skimage.transform.resize(img, (dim, dim))

        #print(img)
        # load label
        lbl_file = data_file['lbl']
        #print("LABEL: ", lbl_file)
        #print()
        #lbl = PIL.Image.open(lbl_file)
        lbl = scipy.misc.imread(lbl_file)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = skimage.transform.resize(lbl, (dim, dim))

        if index % 100 == 0:
            img_name = "images/image" + str(index) + ".jpg"
            skimage.io.imsave(img_name, img)
            lbl_name = "labels/lbl" + str(index) + ".jpg"
            skimage.io.imsave(lbl_name, lbl)



        #lbl[lbl > 0 and lbl < 255] = 0
        # if(len(np.unique(lbl)) > 2):
        #     print()
        #     print()
        #     print()
        #     print(np.unique(lbl))
        #     print(lbl[0].tolist())
        #     print()
        #     print(data_file)
        #     print()
        #     print()
        lbl[lbl > 0] = 1

        #print(np.unique(lbl))


        # print("image: ", img.shape)
        # print("*"*42)
        # print("label: ",lbl.shape)

        # h, w, _ = img.shape
        # top   = random.randint(0, h - new_h)
        # left  = random.randint(0, w - new_w)
        # img   = img[top:top + new_h, left:left + new_w]
        # lbl = lbl[top:top + new_h, left:left + new_w]

        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl


    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)#/255
        # img[0] -= self.means[0]
        # img[1] -= self.means[1]
        # img[2] -= self.means[2]
        img = torch.from_numpy(img.copy()).float()
        lbl = torch.from_numpy(lbl.copy()).long()
        #print(img)#, lbl)
        return img, lbl

    def untransform(self, img, lbl):
        print("utransform!! - will this be effected by resizing")
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        #img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl

#
# def show_batch(batch):
#     img_batch = batch['X']
#     img_batch[:,0,...].add_(means[0])
#     img_batch[:,1,...].add_(means[1])
#     img_batch[:,2,...].add_(means[2])
#     batch_size = len(img_batch)
#
#     grid = utils.make_grid(img_batch)
#     plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))
#
#     plt.title('Batch from dataloader')
#
# if __name__ == "__main__":
#     train_data = SatelliteDataset(phase='train')
#
#     # show a batch
#     batch_size = 4
#     for i in range(batch_size):
#         sample = train_data[i]
#         print(i, sample['X'].size(), sample['Y'].size())
#
#     dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
#
#     for i, batch in enumerate(dataloader):
#         print(i, batch['X'].size(), batch['Y'].size())
#
#         # observe 4th batch
#         if i == 3:
#             plt.figure()
#             show_batch(batch)
#             plt.axis('off')
#             plt.ioff()
#             plt.show()
#             break
