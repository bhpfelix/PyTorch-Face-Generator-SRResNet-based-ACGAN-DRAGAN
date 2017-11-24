# -*- coding: utf-8 -*-

import argparse
import shutil

import numpy as np
import torch
from torch.autograd import Variable, grad
from torch.nn.init import xavier_normal
from torchvision import datasets, transforms
import torchvision.utils as vutils

from data import *
from models import *
from util import *
import time


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


generator = Generator()
generator.apply(weights_init)

discriminator = Discriminator()
discriminator.apply(weights_init)

opt_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

if resume_file:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss_g'] + checkpoint['loss_d']
        generator.load_state_dict(checkpoint['g_state_dict'])
        discriminator.load_state_dict(checkpoint['d_state_dict'])
        opt_g.load_state_dict(checkpoint['g_optimizer'])
        opt_d.load_state_dict(checkpoint['d_optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_file))


criterion = StableBCELoss() # torch.nn.BCEWithLogitsLoss()
X = Variable(torch.FloatTensor(batch_size, 3, imsize, imsize))
z = Variable(torch.FloatTensor(batch_size, z_dim))
tags = Variable(torch.FloatTensor(batch_size, tag_num))
labels = Variable(torch.FloatTensor(batch_size))

if cuda:
    generator.cuda()
    discriminator.cuda()
    criterion.cuda()
    X, z, tags, labels = X.cuda(), z.cuda(), tags.cuda(), labels.cuda()

# Train
best_loss = float('inf')
for epoch in range(start_epoch, max_epochs):
    adjust_learning_rate(opt_g, epoch)
    adjust_learning_rate(opt_d, epoch)

    for batch_idx, (data, target) in enumerate(train_loader):

        X.data.copy_(data)

        # Update discriminator
        # train with real
        tags.data.copy_(target)
        discriminator.zero_grad()
        pred_real, pred_real_tag = discriminator(X)
        labels.data.fill_(1.0)
        loss_d_real_label = criterion(torch.squeeze(pred_real), labels)
        loss_d_real_tag = criterion(pred_real_tag, tags)
        loss_d_real = lambda_adv * loss_d_real_label + loss_d_real_tag
        loss_d_real.backward()

        # train with fake
        z.data.normal_(0, 1)
        tags.data.random_(from=0, to=1) # Continuous
        # tags.data.fill_(0.5)           # Discrete binary string
        # tags.data.bernoulli_()
        rep = torch.cat((z, tags.clone()), 1)
        fake = generator.forward(rep).detach()
        pred_fake, pred_fake_tag = discriminator(fake)
        labels.data.fill_(0.0)
        loss_d_fake_label = criterion(torch.squeeze(pred_fake), labels)
        loss_d_fake_tag = criterion(pred_fake_tag, tags)
        loss_d_fake = lambda_adv * loss_d_fake_label + loss_d_fake_tag
        loss_d_fake.backward()

        # gradient penalty
        shape = [batch_size] + [1]*(X.dim()-1)
        alpha = torch.rand(*shape)
        beta = torch.rand(X.size())
        if cuda:
            alpha = alpha.cuda()
            beta = beta.cuda()
        x_hat = Variable(alpha * X.data + (1 - alpha) * (X.data + 0.5 * X.data.std() * beta), requires_grad=True)
        pred_hat, _ = discriminator(x_hat)
        grad_out = torch.ones(pred_hat.size())
        if cuda:
            grad_out = grad_out.cuda()
        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=grad_out,
                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty.backward()

        loss_d = loss_d_real + loss_d_fake + gradient_penalty
        opt_d.step()

        # Update generator
        generator.zero_grad()
        z.data.normal_(0, 1)
        tags.data.random_(from=0, to=1) # Continuous
        # tags.data.fill_(0.5)           # Discrete binary string
        # tags.data.bernoulli_()
        rep = torch.cat((z, tags.clone()), 1)
        gen = generator(rep)
        pred_gen, pred_gen_tag = discriminator(gen)
        labels.data.fill_(1)

        loss_g_label = criterion(torch.squeeze(pred_gen), labels)

        loss_g_tag = criterion(pred_gen_tag, tags)
        loss_g = lambda_adv * loss_g_label + loss_g_tag
        loss_g.backward()
        opt_g.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_D_Label: %.4f Loss_G_Label: %.4f Loss_D_Tag: %.4f Loss_G_Tag: %.4f'
              % (epoch, max_epochs, batch_idx, len(train_loader),
                 loss_d.data[0], loss_g.data[0], loss_d_real_label.data[0]+loss_d_fake_label.data[0],
                 loss_g_label.data[0], loss_d_real_tag.data[0]+loss_d_fake_tag.data[0],
                 loss_g_tag.data[0]))

        if batch_idx % 100 == 0:
            vutils.save_image(data,
                    'samples/real_samples.png')
            fake = generator(rep)
            vutils.save_image(fake.data.view(batch_size, 3, imsize, imsize),
                    'samples/fake_samples_epoch_%03d.png' % epoch)

            is_best = False

            ### Should not be able to define best model..
            # total_loss = loss_d.data[0] + loss_g.data[0]
            # if total_loss < best_loss:
            #     best_loss = total_loss
            #     is_best = True

            save_checkpoint({
                'epoch': epoch,
                'arch': 'ACGAN-SRResNet-DRAGAN',
                'g_state_dict': generator.state_dict(),
                'd_state_dict': discriminator.state_dict(),
                'loss_g': loss_g.data[0],
                'loss_d': loss_d.data[0],
                'g_optimizer' : opt_g.state_dict(),
                'd_optimizer' : opt_d.state_dict(),
            }, is_best, 'models/Epoch: %03d.pt' % (epoch))