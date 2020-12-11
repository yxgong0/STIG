import torch
import numpy as np
import torch.utils.data as data
from os import listdir
import os
from PIL import Image
import random
from utils.utils import *


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def default_loader(path):
    return Image.open(path).convert('RGB')


def ToTensor(pic):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backard compability
        return img.float().div(255)
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


class TestDataset(data.Dataset):
    def __init__(self, data_path='', size_w=256, size_h=256):
        super(TestDataset, self).__init__()
        self.image_list = [x for x in listdir(data_path) if is_image_file(x)]
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h

    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.image_list[index])
        try:
            img = default_loader(img_path)
        except IOError:
            img = default_loader('samples/100_0.jpg')
            self.image_list[index] = 'broken'

        name = self.image_list[index]

        img = img.resize((self.size_w, self.size_h), Image.BILINEAR)
        img = process_mask(img)
        img = ToTensor(img)

        img = img.mul_(2).add_(-1)
        return img, name

    def __len__(self):
        return len(self.image_list)


class TrainDataset(data.Dataset):
    def __init__(self, data_path='', size_w=128, size_h=64):
        super(TrainDataset, self).__init__()
        self.image_list = [x for x in listdir(data_path + 'image/') if is_image_file(x)]
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h

    def __getitem__(self, index):
        broken_instead = 'samples/100_0.jpg'

        mask_path = os.path.join(self.data_path + 'mask/', self.image_list[index])
        try:
            mask = default_loader(mask_path)
        except IOError:
            mask = default_loader(broken_instead)
            self.image_list[index] = 'broken'

        img_path = os.path.join(self.data_path + 'image/', self.image_list[index])
        try:
            img = default_loader(img_path)
        except IOError:
            img = default_loader(broken_instead)
            self.image_list[index] = 'broken'

        mask_dilated_path = os.path.join(self.data_path + 'masktr/', self.image_list[index])
        try:
            mask_dilated = default_loader(mask_dilated_path)
        except IOError:
            mask_dilated = default_loader(broken_instead)
            self.image_list[index] = 'broken'

        mask_transition = mask_dilated.copy()
        for s in range(0, mask_transition.size[0]):
            for t in range(0, mask_transition.size[1]):
                pixel = mask_transition.getpixel((s,t))
                if pixel[0] >= 128 and pixel[1] >= 128 and pixel[2] >= 128:
                    new_pixel = img.getpixel((s, t))
                    mask_transition.putpixel((s, t), new_pixel)
                else:
                    new_pixel = mask.getpixel((s, t))
                    mask_transition.putpixel((s, t), new_pixel)

        img = img.resize((self.size_w, self.size_h), Image.BILINEAR)
        mask = mask.resize((self.size_w, self.size_h), Image.BILINEAR)
        mask_transition = mask_transition.resize((self.size_w, self.size_h), Image.BILINEAR)
        mask = process_mask(mask)

        img = ToTensor(img)
        mask = ToTensor(mask)
        mask_transition = ToTensor(mask_transition)

        img = img.mul_(2).add_(-1)
        mask = mask.mul_(2).add_(-1)
        mask_transition = mask_transition.mul_(2).add_(-1)
        return img, mask, mask_transition, self.image_list[index]

    def __len__(self):
        return len(self.image_list)
