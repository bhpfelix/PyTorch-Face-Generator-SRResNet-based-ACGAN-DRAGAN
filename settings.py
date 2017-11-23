# -*- coding: utf-8 -*-

import torch

DATA_PATH = '/home/xander/data/img_align_celeba/'

resume_file = 'models/Epoch: 018.pt'
cuda = torch.cuda.is_available()
batch_size = 48
z_dim = 128
tag_num = 19
imsize = 128
start_epoch = 0
max_epochs = 100000
lambda_adv = tag_num
lambda_gp = 0.5
learning_rate = 0.0002
