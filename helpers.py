import collections
import os.path as osp
import pickle
import random

import PIL.Image
import PIL.ImageOps
import matplotlib.pyplot as plt
# from __future__ import division
import numpy as np
import torch
import torchvision
import torchvision.utils
from PIL import Image
from sklearn.datasets import fetch_lfw_pairs
from torch.utils import data
from torch.utils.data import DataLoader, Dataset


def imshow(img, text=None, should_save=False):
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


class Config():
    root = "/home/jameson/face_similarity/siamese_density_networks/data/att_faces/"
    training_dir = root + "training/"
    testing_dir = root + "testing/"
    train_batch_size = 64
    train_number_epochs = 100


def visualize_faces(siamese_dataset):
    vis_dataloader = DataLoader(siamese_dataset,
                                shuffle=True,
                                num_workers=8,
                                batch_size=8)
    dataiter = iter(vis_dataloader)
    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True, colour=False):
        self.colour = colour
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.colour:
            img0 = img0.convert("RGB")
            img1 = img1.convert("RGB")

        else:
            img0 = img0.convert("L")
            img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class WildFacesDataset(Dataset):

    def __init__(self, train=True, color=False, funneled=True, resize=1.0, transform=None, should_invert=True):
        splitName = 'train' if train else 'test'
        self.data = fetch_lfw_pairs(subset=splitName, funneled=funneled, resize=resize, color=color)
        print("Wild Faces")
        print(self.data.pairs.shape)
        self.image_len = len(self.data.target)
        self.train = train
        self.transform = transform
        self.should_invert = should_invert
        self.color = color

    def __getitem__(self, index):
        img0, img1 = np.expand_dims(self.data.pairs[:, 0, :, :], 1), np.expand_dims(self.data.pairs[:, 1, :, :], 1)
        img0 = selftorch.from_numpy(img0)
        img1 = torch.from_numpy(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        targets = torch.from_numpy(self.data.target.astype(np.float32))
        return (img0, img1, targets)

    def __len__(self):
        return (self.image_len)


class DemoFaceDataset(data.Dataset):
    '''
        Dataset subclass for demonstrating how to load images in PyTorch.

    '''

    # -----------------------------------------------------------------------------
    def __init__(self, root, split='train', set='tiny', im_size=250):
        # -----------------------------------------------------------------------------
        '''
            Parameters
            ----------
            root        -   Path to root of ImageNet dataset
            split       -   Either 'train' or 'val'
            set         -   Can be 'full', 'small' or 'tiny' (5 images)
        '''
        self.root = root  # E.g. '.../ImageNet/images' or '.../vgg-face/images'
        self.split = split
        self.files = collections.defaultdict(list)
        self.im_size = im_size  # scale image to im_size x im_size
        self.set = set

        if set == 'small':
            raise NotImplementedError()

        elif set == 'tiny':
            # DEBUG: 5 images
            files_list = osp.join(root, 'tiny_face_' + self.split + '.txt')

        elif set == 'full':
            raise NotImplementedError()

        else:
            raise ValueError('Valid sets: `full`, `small`, `tiny`.')

        assert osp.exists(files_list), 'File does not exist: %s' % files_list

        imfn = []
        with open(files_list, 'r') as ftrain:
            for line in ftrain:
                imfn.append(osp.join(root, line.strip()))
        self.files[split] = imfn

    # -----------------------------------------------------------------------------
    def __len__(self):
        # -----------------------------------------------------------------------------
        return len(self.files[self.split])

    # -----------------------------------------------------------------------------
    def __getitem__(self, index):
        # -----------------------------------------------------------------------------
        img_file = self.files[self.split][index]
        img = PIL.Image.open(img_file)

        # HACK: for non-RGB images - 4-channel CMYK or 1-channel grayscale
        if len(img.getbands()) != 3:
            while len(img.getbands()) != 3:
                index -= 1
                img_file = self.files[self.split][index]  # if -1, wrap-around
                img = PIL.Image.open(img_file)

        if self.im_size > 0:
            # Scales image to a square of default size 250x250
            scaled_dim = (self.im_size.astype(np.int32),
                          self.im_size.astype(np.int32))
            img = img.resize(scaled_dim, PIL.Image.BILINEAR)

        label = 1  # TODO: read in a class label for each image

        img = np.array(img, dtype=np.uint8)
        im_out = torch.from_numpy(im_out).float()
        im_out = im_out.permute(2, 0, 1)  # C x H x W

        return im_out, label


class LFWDataset(data.Dataset):
    '''
        Dataset subclass for loading LFW images in PyTorch.
        This returns multiple images in a batch.
    '''

    def __init__(self, path_list, issame_list, transforms, split='test'):
        '''
            Parameters
            ----------
            path_list    -   List of full path-names to LFW images
        '''
        self.files = collections.defaultdict(list)
        self.split = split
        self.files[split] = path_list
        self.pair_label = issame_list
        self.transforms = transforms

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_file = self.files[self.split][index]
        img = PIL.Image.open(img_file)
        if DEBUG:
            print img_file
        im_out = self.transforms(img)
        return im_out


class IJBADataset(data.Dataset):
    '''
        Dataset subclass for loading IJB-A images in PyTorch.
        This returns multiple images in a batch.
        Path_list -- full paths to cropped images saved as <sighting_id>.jpg 
    '''

    def __init__(self, path_list, transforms, split=1):
        '''
            Parameters
            ----------
            path_list    -   List of full path-names to IJB-A images of one split  
        '''
        self.files = collections.defaultdict(list)
        self.split = split
        self.files[split] = path_list
        self.transforms = transforms

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_file = self.files[self.split][index]
        img = PIL.Image.open(img_file)
        if not img.mode == 'RGB':
            img = img.convert('RGB')
        if DEBUG:
            print img_file
        im_out = self.transforms(img)
        return im_out


class DemoFaceDataset(data.Dataset):
    '''
        Dataset subclass for demonstrating how to load images in PyTorch.

    '''

    # -----------------------------------------------------------------------------
    def __init__(self, root, split='train', set='tiny', im_size=250):
        # -----------------------------------------------------------------------------
        '''
            Parameters
            ----------
            root        -   Path to root of ImageNet dataset
            split       -   Either 'train' or 'val'
            set         -   Can be 'full', 'small' or 'tiny' (5 images)
        '''
        self.root = root  # E.g. '.../ImageNet/images' or '.../vgg-face/images'
        self.split = split
        self.files = collections.defaultdict(list)
        self.im_size = im_size  # scale image to im_size x im_size
        self.set = set

        if set == 'small':
            raise NotImplementedError()

        elif set == 'tiny':
            # DEBUG: 5 images
            files_list = osp.join(root, 'tiny_face_' + self.split + '.txt')

        elif set == 'full':
            raise NotImplementedError()

        else:
            raise ValueError('Valid sets: `full`, `small`, `tiny`.')

        assert osp.exists(files_list), 'File does not exist: %s' % files_list

        imfn = []
        with open(files_list, 'r') as ftrain:
            for line in ftrain:
                imfn.append(osp.join(root, line.strip()))
        self.files[split] = imfn

    # -----------------------------------------------------------------------------
    def __len__(self):
        # -----------------------------------------------------------------------------
        return len(self.files[self.split])

    # -----------------------------------------------------------------------------
    def __getitem__(self, index):
        # -----------------------------------------------------------------------------
        img_file = self.files[self.split][index]
        img = PIL.Image.open(img_file)

        # HACK: for non-RGB images - 4-channel CMYK or 1-channel grayscale
        if len(img.getbands()) != 3:
            while len(img.getbands()) != 3:
                index -= 1
                img_file = self.files[self.split][index]  # if -1, wrap-around
                img = PIL.Image.open(img_file)

        if self.im_size > 0:
            # Scales image to a square of default size 250x250
            scaled_dim = (self.im_size.astype(np.int32),
                          self.im_size.astype(np.int32))
            img = img.resize(scaled_dim, PIL.Image.BILINEAR)

        label = 1  # TODO: read in a class label for each image

        img = np.array(img, dtype=np.uint8)
        im_out = torch.from_numpy(im_out).float()
        im_out = im_out.permute(2, 0, 1)  # C x H x W

        return im_out, label


class LFWDataset(data.Dataset):
    '''
        Dataset subclass for loading LFW images in PyTorch.
        This returns multiple images in a batch.
    '''

    def __init__(self, path_list, issame_list, transforms, split='test'):
        '''
            Parameters
            ----------
            path_list    -   List of full path-names to LFW images
        '''
        self.files = collections.defaultdict(list)
        self.split = split
        self.files[split] = path_list
        self.pair_label = issame_list
        self.transforms = transforms

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_file = self.files[self.split][index]
        img = PIL.Image.open(img_file)
        if DEBUG:
            print img_file
        im_out = self.transforms(img)
        return im_out


class IJBADataset(data.Dataset):
    '''
        Dataset subclass for loading IJB-A images in PyTorch.
        This returns multiple images in a batch.
        Path_list -- full paths to cropped images saved as <sighting_id>.jpg 
    '''

    def __init__(self, path_list, transforms, split=1):
        '''
            Parameters
            ----------
            path_list    -   List of full path-names to IJB-A images of one split  
        '''
        self.files = collections.defaultdict(list)
        self.split = split
        self.files[split] = path_list
        self.transforms = transforms

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_file = self.files[self.split][index]
        img = PIL.Image.open(img_file)
        if not img.mode == 'RGB':
            img = img.convert('RGB')
        if DEBUG:
            print img_file
        im_out = self.transforms(img)
        return im_out


def save_obj(obj, name):
    with open('results/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('results/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_model(name, model):
    with open('results/' + name + '_model', 'wb') as f:
        torch.save(model.state_dict(), f)


def load_model(model_path, model):
    checkpoint = torch.load(model_path)
    return model.load_state_dict(checkpoint['state_dict'])


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename + '.tar')


def load_checkpoint(model, optimizer, args):
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            args.start_epoch = checkpoint['epoch']
            args.best_1oss = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    return (model, optimizer, args)
