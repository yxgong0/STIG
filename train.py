import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from model.Discriminator import Discriminator
from model.Generator import Generator
from utils.dataset import TrainDataset

parser = argparse.ArgumentParser(description='train pix2pix model')
parser.add_argument('--batch_size', type=int, default=64, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--size_w', type=int, default=128, help='scale image to this size')
parser.add_argument('--size_h', type=int, default=64, help='scale image to this size')
parser.add_argument('--input_nc', type=int, default=3, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=3, help='channel number of output image')
parser.add_argument('--lamb', type=int, default=1, help='weight on L1 term in objective')
parser.add_argument('--netG', type=str, default='', help='path to pre-trained netG')
parser.add_argument('--netD', type=str, default='', help='path to pre-trained netD')
parser.add_argument('--data_path', default='samples/train_dir/', help='path to training images')
parser.add_argument('--outf', default='results/', help='folder to output images and model checkpoints')
opt = parser.parse_args()
opt.cuda = True
print(opt)


def makedirs_(outf=''):
    items = outf.split('/')
    for subfolder_number in range(0, items.__len__(), 1):
        dir_name = ''
        for number in range(0, subfolder_number, 1):
            dir_name += '/'
            dir_name += items[number]
        try:
            os.makedirs(outf)
        except OSError:
            pass
    try:
        os.makedirs(outf + 'model/')
    except OSError:
        pass


makedirs_(opt.outf)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

###########   DATASET   ###########
train_dataset = TrainDataset(opt.data_path, opt.size_w, opt.size_h)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=opt.batch_size,
                                           shuffle=True,
                                           num_workers=6)


###########   MODEL   ###########
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


generator = Generator(opt.input_nc, opt.output_nc, opt.ngf)
discriminator = Discriminator(opt.input_nc, opt.output_nc, opt.ndf)

netD1 = discriminator
netD2 = discriminator
netG1 = generator
netG2 = generator

if opt.netG != '':
    netG1.load_state_dict(torch.load(opt.netG))
    netG2.load_state_dict(torch.load(opt.netG))
    netD1.load_state_dict(torch.load(opt.netD))
    netD2.load_state_dict(torch.load(opt.netD))
if opt.cuda:
    netD1.cuda()
    netD2.cuda()
    netG1.cuda()
    netG2.cuda()

if opt.netG == '':
    netG1.apply(weights_init)
    netG2.apply(weights_init)
    netD1.apply(weights_init)
    netD2.apply(weights_init)
print(netD1)
print(netG1)

###########   LOSS & OPTIMIZER   ##########
criterion1 = nn.BCELoss()
criterion2 = nn.BCELoss()
criterionL11 = nn.L1Loss()
criterionL12 = nn.L1Loss()
optimizerG1 = torch.optim.Adam(netG1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG2 = torch.optim.Adam(netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD1 = torch.optim.SGD(netD1.parameters(), lr=opt.lr/2, momentum=0.9)
optimizerD2 = torch.optim.SGD(netD2.parameters(), lr=opt.lr/2, momentum=0.9)

###########   GLOBAL VARIABLES   ###########
real_mask = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
real_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
real_masktr = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
label = torch.FloatTensor(opt.batch_size)

if opt.cuda:
    real_mask = real_mask.cuda()
    real_image = real_image.cuda()
    real_masktr = real_masktr.cuda()
    label = label.cuda()

real_label = 1
fake_label = 0

if __name__ == '__main__':
    ########### Training   ###########
    netD1.train()
    netD2.train()
    netG1.train()
    netG2.train()
    for epoch in range(1, opt.epochs+1):
        loader = iter(train_loader)
        for i in range(0, train_dataset.__len__(), opt.batch_size):
            image, mask, masktr, name = loader.next()
            if name == 'broken':
                continue

            real_mask.resize_(mask.size()).copy_(mask)
            real_image.resize_(image.size()).copy_(image)
            real_masktr.resize_(masktr.size()).copy_(masktr)
            ########### fDx1 ###########
            netD1.zero_grad()
            real_mask_masktr = torch.cat((real_mask, real_masktr), 1)
            output1 = netD1(real_mask_masktr)
            label.resize_(output1.size())
            label.fill_(real_label)
            errD1_real = criterion1(output1, label)
            errD1_real.backward()
            ########### real data trained
            fake_masktr = netG1(real_mask)
            label.data.fill_(fake_label)
            fake_mask_masktr = torch.cat((real_mask, fake_masktr), 1)
            output1 = netD1(fake_mask_masktr.detach())
            errD1_fake = criterion1(output1, label)
            errD1_fake.backward()
            ########### fake data trained
            errD1 = (errD1_fake + errD1_real)/2
            optimizerD1.step()
            ########### fGx1 ##########
            netG1.zero_grad()
            label.fill_(real_label)
            output1 = netD1(fake_mask_masktr)
            errGAN1 = criterion1(output1, label)
            errL11 = criterionL11(fake_masktr, real_masktr)
            errG1 = errGAN1 + opt.lamb * errL11
            errG1.backward()
            optimizerG1.step()

            ########### fDx2 ##########
            netD2.zero_grad()
            real_masktr_image = torch.cat((real_masktr, real_image), 1)
            output2 = netD2(real_masktr_image)
            label.fill_(real_label)
            errD2_real = criterion2(output2, label)
            errD2_real.backward()
            ########## real data trained
            fake_image = netG2(real_masktr)
            fake_masktr_image = torch.cat((real_masktr, fake_image), 1)
            output2 = netD2(fake_masktr_image.detach())
            label.fill_(fake_label)
            errD2_fake = criterion2(output2, label)
            errD2_fake.backward()
            ########## fake data trained
            errD2 = (errD2_fake + errD2_real) / 2
            optimizerD2.step()
            ########### fGx2 ###########
            netG2.zero_grad()
            output2 = netD2(fake_masktr_image)
            label.fill_(real_label)
            errGAN2 = criterion2(output2, label)
            errL12 = criterionL12(fake_image, real_image)
            errG2 = errGAN2 + opt.lamb*errL12
            errG2.backward()
            optimizerG2.step()

            ########### Logging ##########
            if i % 2 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f %.4f Loss_GAN: %.4f %.4f Loss_L1: %.4f %.4f Loss_G: %.4f %.4f'
                      % (epoch, opt.epochs, i, len(train_loader), errD1.item(), errD2.item(),
                         errGAN1.item(), errGAN2.item(), errL11.item(), errL12.item(), errG1.item(), errG2.item()))

            if i % 200 == 0:
                vutils.save_image(fake_image.data,
                                  opt.outf + 'fake_samples=_epoch_%03d_%03d.png' % (epoch, i),
                                  normalize=True)
            if i % 200 == 0:
                vutils.save_image(fake_masktr.data,
                                  opt.outf + 'fake_masktr_samples_epoch_%03d_%03d.png' % (epoch, i),
                                  normalize=True)

        if epoch % 20 == 0:
            torch.save(netG1.state_dict(), '%s/model/netG1_%s.pth' % (opt.outf, str(epoch)))
            torch.save(netD1.state_dict(), '%s/model/netD1_%s.pth' % (opt.outf, str(epoch)))
            torch.save(netG2.state_dict(), '%s/model/netG2_%s.pth' % (opt.outf, str(epoch)))
            torch.save(netD2.state_dict(), '%s/model/netD2_%s.pth' % (opt.outf, str(epoch)))

    torch.save(netG1.state_dict(), '%s/model/netG1.pth' % opt.outf)
    torch.save(netD1.state_dict(), '%s/model/netD1.pth' % opt.outf)
    torch.save(netG2.state_dict(), '%s/model/netG2.pth' % opt.outf)
    torch.save(netD2.state_dict(), '%s/model/netD2.pth' % opt.outf)
