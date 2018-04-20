"""
Sequential FER datasets loading and processing

author: Yajue Yang
"""

from __future__ import print_function
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import matplotlib.pyplot as plt

SAMPLE_INPUT = 'video'
SAMPLE_TARGET = 'label'


def get_video_dir_paths(root_dir):
    video_dir_paths = []
    for dir in sorted(os.listdir(root_dir)):
        # print(dir)
        full_dir = os.path.join(root_dir, dir)
        for sub_dir in sorted(os.listdir(full_dir)):
            if sub_dir[0] != '.':
                # print(sub_dir)
                full_sub_dir = os.path.join(full_dir, sub_dir)
                video_dir_paths.append(full_sub_dir)
                # for img in sorted(glob.glob(os.path.join(full_sub_dir, '*.png'))):
                #     print(img)
        # print('\n')
    return video_dir_paths


def get_label_dir_paths(root_dir):
    label_dir_paths = []
    for dir in sorted(os.listdir(root_dir)):
        # print(dir)
        full_dir = os.path.join(root_dir, dir)
        for sub_dir in sorted(os.listdir(full_dir)):
            if sub_dir[0] != '.':
                # print(sub_dir)
                full_sub_dir = os.path.join(full_dir, sub_dir)
                if glob.glob(os.path.join(full_sub_dir, '*.txt')):
                    label_dir_paths.append(full_sub_dir)
    return label_dir_paths


def match_ckvl_dir_paths(vd_paths, ld_paths):

    vd_check_idx = 0

    for ld in ld_paths:

        ld_words = ld.split('/')

        vd_words = vd_paths[vd_check_idx].split('/')

        while vd_words[-1] != ld_words[-1] or vd_words[-2] != ld_words[-2]:
            vd_paths.remove(vd_paths[vd_check_idx])
            vd_words = vd_paths[vd_check_idx].split('/')

        vd_check_idx += 1


def get_ck_data(v_root_dir, l_root_dir):
    video_dir_paths = get_video_dir_paths(v_root_dir)
    label_dir_paths = get_label_dir_paths(l_root_dir)

    match_ckvl_dir_paths(video_dir_paths, label_dir_paths)

    return video_dir_paths, label_dir_paths


def calc_img_dataset_mean_std(vd_paths, transform):
    """
    Calculate statistics (mean and standard deviation) of the given image dataset
    :param vd_paths: video (a sequence of images) directory paths
    :param transform: transform functions of the original images
    :return: mean and std
    """

    all_images = []

    for path in vd_paths:
        img_names = glob.glob(os.path.join(path, '*.png'))

        for img in img_names:
            img = Image.open(img)
            img = transform(img)
            all_images.append(img)

    if isinstance(all_images[0], torch.FloatTensor):
        # print(all_images[0])
        all_images = torch.stack(all_images)
        img_mean = torch.mean(all_images, 0)
        img_std = torch.std(all_images, 0)

        return img_mean, img_std

    # todo: numpy array images...


class ImgMeanStdNormalization(object):
    """
    Normalize an image by subtracting the mean image and then dividing the standard deviation
    """
    def __init__(self, mean_img, std):
        self.mean_img = mean_img
        self.std = std

    def __call__(self, image):
        if isinstance(image, torch.FloatTensor):
            image -= self.mean_img
            image /= self.std
            return image


class SFERDataset(Dataset):

    def __init__(self, video_dir_paths, label_dir_paths, transform=None):
        self.video_dir_paths = video_dir_paths
        self.label_dir_paths = label_dir_paths

        assert self.vl_matched()

        self.transform = transform

    def vl_matched(self):

        all_matched = True

        for vd, ld in zip(self.video_dir_paths, self.label_dir_paths):
            vd_words = vd.split('/')
            ld_words = ld.split('/')

            if vd_words[-1] != ld_words[-1] or vd_words[-2] != ld_words[-2]:
                all_matched = False
                break

        return all_matched

    def __len__(self):
        return len(self.video_dir_paths)

    def __getitem__(self, idx):

        # obtain the sequence of image names
        v_imgs = []

        img_names = sorted(glob.glob(os.path.join(self.video_dir_paths[idx], '*.png')))

        for img in img_names:
            img = Image.open(img)
            if self.transform:
                img = self.transform(img)
            v_imgs.append(img)

        # obtain the label
        label_files = glob.glob(os.path.join(self.label_dir_paths[idx], '*.txt'))
        print(self.label_dir_paths[idx])
        print('number of frame: ', len(img_names))
        # print(label_files)
        em = 0
        with open(label_files[0]) as f:
            for line in f:
                line = line.lstrip(' ')
                line = line.split('.')
                em = int(line[0])

        sample = {SAMPLE_INPUT: torch.stack(v_imgs), SAMPLE_TARGET: em}

        return sample


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class SFERPadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a IntTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[SAMPLE_INPUT].size()[0], batch))
        # print(max_len)

        # pad according to max_len
        batch = list(map(lambda x: {SAMPLE_INPUT: pad_tensor(x[SAMPLE_INPUT], max_len, self.dim), SAMPLE_TARGET: x[SAMPLE_TARGET]}, batch))

        # stack all
        xs = torch.stack([sample[SAMPLE_INPUT] for sample in batch], dim=0)
        ys = torch.IntTensor([sample[SAMPLE_TARGET] for sample in batch])

        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)


if __name__ == '__main__':

    video_root_dir = r'/home/young/cv_project/cohn-kanade-images'

    label_root_dir = r'/home/young/cv_project/Emotion'

    video_dir_paths, label_dir_paths = get_ck_data(video_root_dir, label_root_dir)

    img_size = (320, 240)
    composed_tf = transforms.Compose([transforms.Grayscale(), transforms.Resize(img_size), transforms.ToTensor()])

    img_mean, img_std = calc_img_dataset_mean_std(video_dir_paths, composed_tf)

    dataset_tf = transforms.Compose([transforms.Grayscale(), transforms.Resize(img_size), transforms.ToTensor(),
                                     ImgMeanStdNormalization(img_mean, img_std)])

    sfer_dataset = SFERDataset(video_dir_paths, label_dir_paths, transform=dataset_tf)

    sample = sfer_dataset.__getitem__(len(video_dir_paths) - 1)

    v_imgs, label = sample[SAMPLE_INPUT], sample[SAMPLE_TARGET]

    print(label)

    print(v_imgs[0].size())
    print(v_imgs[0])
    print(torch.min(v_imgs[0]))
    print(torch.max(v_imgs[0]))
