# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image

from settings import *

class AttrDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.attr_list = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.attr_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.attr_list.ix[idx, 0])
        image = Image.open(img_name)
        attrs = self.attr_list.ix[idx, 1:].as_matrix().astype('float')
        attrs = torch.FloatTensor(attrs)

        if self.transform:
            image = self.transform(image)

        return image, attrs


face_transform = transforms.Compose([
           transforms.Scale(128),
           transforms.CenterCrop(size=128),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
       ])


print('Preparing dataset A...')
A_train_dataset = AttrDataset('face_attr.csv', DATA_PATH, transform=face_transform)
train_loader = DataLoader(A_train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)