import torchvision.datasets as dset
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.misc
import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader


########## ----------------------------------- LOADERS -------------------------------------- ####


def get_att(args):
    folder_dataset_train = dset.ImageFolder(root=args.training_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_train,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ]), should_invert=False)
    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=8,
                                  batch_size=args.train_batch_size)

    folder_dataset_test = dset.ImageFolder(root=args.testing_dir)
    siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                 transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                               transforms.ToTensor()
                                                                               ]), should_invert=False)

    test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=args.train_batch_size, shuffle=False)
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)
    return (train_dataloader, test_dataloader, dataiter)


########## ----- from local data: this works but should be using pairs.txt for training loader and pairsDevTest for testing ----- ####

def get_funneled_wild(args):
    folder_dataset_train = dset.ImageFolder(
        root="/home/jameson/face_similarity/siamese_density_networks/data/LFW_DIR/lfw-deepfunneled/")
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_train,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ]), should_invert=False, colour=True)
    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=8,
                                  batch_size=args.train_batch_size)

    folder_dataset_test = dset.ImageFolder(root=args.testing_dir)
    siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                 transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                               transforms.ToTensor()
                                                                               ]), should_invert=False, colour=True)

    test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=args.train_batch_size, shuffle=False)
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)
    return (train_dataloader, test_dataloader, dataiter)


def get_funneled_wild_new(args, root="/home/jameson/face_similarity/siamese_density_networks/data/LFW_DIR/"):
    train_dataloader = torch.utils.data.DataLoader(train_ImageList(fileList=root + "train.txt",
                                                                   transform=transforms.Compose(
                                                                       [transforms.Resize((100, 100)),
                                                                        transforms.ToTensor()])),
                                                   shuffle=False, num_workers=8, batch_size=args.train_batch_size)

    test_dataloader = torch.utils.data.DataLoader(test_ImageList(fileList=root + "test.txt",
                                                                 transform=transforms.Compose(
                                                                     [transforms.Resize((100, 100)),
                                                                      transforms.ToTensor()])),
                                                  shuffle=False, num_workers=8, batch_size=8)

    dataiter = iter(test_dataloader)
    # x0,_,_ = next(dataiter)

    return (train_dataloader, test_dataloader, dataiter)


######  ----------- uses sklearn data: error since transform not working on numpy array so cannot batch it ------------ ##########


def get_wild(args):
    train_dataset = WildFacesDataset(train=True, color=args.color, resize=1.0, funneled=args.funneled,
                                     transform=transforms.Compose([transforms.Resize((100, 100))]), should_invert=False)

    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=args.train_batch_size)

    test_dataset = WildFacesDataset(train=False, color=args.color, resize=1.0, funneled=args.funneled,
                                    transform=transforms.Compose([transforms.Resize((100, 100))]), should_invert=False)

    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=8, batch_size=args.train_batch_size)
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)
    return (train_dataloader, test_dataloader, dataiter)


######  ----------- not used because not working ------------ ##########


def new_get_wild(args):
    file_ext = 'jpg'  # observe, no '.' before jpg
    num_class = 8631
    root_path = "/home/jameson/face_similarity/siamese_density_networks/data/LFW_DIR/"
    # if args.fold == 0:
    train_pairs_path = root_path + 'pairsDevTest.txt'
    # else:
    test_pairs_path = root_path + 'pairs.txt'

    train_pairs = utils.read_pairs(train_pairs_path)
    test_pairs = utils.read_pairs(test_pairs_path)

    train_path_list, issame_train_list = utils.get_paths(args.dataset_path, train_pairs, file_ext)
    test_path_list, issame_test_list = utils.get_paths(args.dataset_path, test_pairs, file_ext)

    # Define data transforms
    RGB_MEAN = [0.485, 0.456, 0.406]
    RGB_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Scale((250, 250)),  # make 250x250
        transforms.CenterCrop(150),  # then take 150x150 center crop
        transforms.Scale((224, 224)),  # resized to the network's required input size
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN,
                             std=RGB_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Scale((250, 250)),  # make 250x250
        transforms.CenterCrop(150),  # then take 150x150 center crop
        transforms.Scale((224, 224)),  # resized to the network's required input size
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN,
                             std=RGB_STD),
    ])

    train_dataset = WildFacesDataset(train=True, color=args.color,
                                     transform=transforms.Compose([transforms.Resize((50, 50)),
                                                                   transforms.ToTensor()
                                                                   ]), should_invert=False)

    test_dataset = WildFacesDataset(train=False, color=args.color,
                                    transform=transforms.Compose([transforms.Resize((50, 50)),
                                                                  transforms.ToTensor()
                                                                  ]), should_invert=False)

    # train_dataset = WildFacesDataset(train = True, color=args.color,  transform=train_transform,transforms.ToTensor() ]), should_invert=False)
    # test_dataset = WildFacesDataset(train = False, color=args.color, transform=train_transform,transforms.ToTensor()]), should_invert=False)

    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        data_loader.LFWDataset(
            train_path_list, issame_train_list, train_transform),
        batch_size=args.train_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        data_loader.LFWDataset(
            test_path_list, issame_test_list, test_transform),
        batch_size=args.train_batch_size, shuffle=False)

    dataiter = iter(test_loader)
    # x0,_,_ = next(dataiter)
    print(train_loader)
    print(test_loader)
    print(dataiter)
    return (train_loader, test_loader, dataiter)


# https://github.com/Jin-Linhao/Siamese_lfw_pytorch/blob/master/scripts/siamese_lfw_train.py
######  ----------- loading so to use the splits specified in the train and test files ------------ ##########

def imshow(img, text, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


def train_loader(path, augment=False):
    img = Image.open(path)
    if augment:
        pix = np.array(img)
        pix_aug = img_augmentation(pix)
        img = Image.fromarray(np.uint8(pix_aug))
    # print pix
    return img


def test_loader(path):
    img = Image.open(path)
    return img


def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgshortList = []
            imgPath1, imgPath2, label = line.strip().split(' ')

            imgshortList.append(imgPath1)
            imgshortList.append(imgPath2)
            imgshortList.append(label)
            imgList.append(imgshortList)
    return imgList


def img_augmentation(img):
    if random.random() > 0.7:

        h, w, c = np.shape(img)
        # scale
        if random.random() > 0.5:
            s = (random.random() - 0.5) / 1.7 + 1
            img = scipy.misc.imresize(img, (int(h * s), int(w * s)))
        # translation
        if random.random() > 0.5:
            img = scipy.ndimage.shift(img, (int(random.random() * 20 - 10), int(random.random() * 20 - 10), 0))
        # rotation
        if random.random() > 0.5:
            img = scipy.ndimage.rotate(img, random.random() * 60 - 30)
        # flipping
        if random.random() > 0.5:
            img = np.flip(img, 1)
        # crop and padding
        h_c, w_c = img.shape[:2]
        if h_c > h:
            top = int(h_c / 2 - h / 2)
            left = int(w_c / 2 - w / 2)
            img_out = img[top: top + h, left: left + w]
        else:
            pad_size = int((h - h_c) / 2)
            pads = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
            img_out = np.pad(np.array(img), pads, 'constant', constant_values=0)
    else:
        img_out = img
    # print np.shape(img_out)
    return img_out


class train_ImageList(data.Dataset):
    def __init__(self, fileList, transform=None, list_reader=default_list_reader, train_loader=train_loader):
        # self.root      = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.train_loader = train_loader

    def __getitem__(self, index):
        lfw_path = "/home/jameson/face_similarity/siamese_density_networks/data/LFW_DIR/lfw-deepfunneled"
        final = []
        [imgPath1, imgPath2, target] = self.imgList[index]
        img1 = self.train_loader(os.path.join(lfw_path, imgPath1))
        img2 = self.train_loader(os.path.join(lfw_path, imgPath2))

        #
        # img2 = self.img_augmentation(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([target], dtype=np.float32))

    def __len__(self):
        return len(self.imgList)


class test_ImageList(data.Dataset):
    def __init__(self, fileList, transform=None, list_reader=default_list_reader, test_loader=test_loader):
        # self.root      = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.test_loader = test_loader

    def __getitem__(self, index):
        lfw_path = "/home/jameson/face_similarity/siamese_density_networks/data/LFW_DIR/lfw-deepfunneled"
        final = []
        [imgPath1, imgPath2, target] = self.imgList[index]
        img1 = self.test_loader(os.path.join(lfw_path, imgPath1))
        img2 = self.test_loader(os.path.join(lfw_path, imgPath2))

        #
        # img2 = self.img_augmentation(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([target], dtype=np.float32))

    def __len__(self):
        return len(self.imgList)
