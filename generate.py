import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from model.Generator import Generator
from utils.dataset import TestDataset

parser = argparse.ArgumentParser(description='test pix2pix model')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG1', default='results/model/netG1.pth', help="path to netG (to continue training)")
parser.add_argument('--netG2', default='results/model/netG2.pth', help="path to netG (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--size_w', type=int, default=128, help='scale image to this size')
parser.add_argument('--size_h', type=int, default=64, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=128, help='random crop image to this size')
parser.add_argument('--input_nc', type=int, default=3, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=3, help='channel number of output image')
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--data_path', default='samples/test_dir/', help='path to training images')
parser.add_argument('--outf', default='results/generated/', help='folder to output images and model checkpoints')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--imgNum', type=int, default=1053, help='How many images to generate?')

opt = parser.parse_args()
opt.cuda = True
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

###########   Load netG   ###########
netG1 = Generator(opt.input_nc, opt.output_nc, opt.ngf)
netG1.load_state_dict(torch.load(opt.netG1))
netG2 = Generator(opt.input_nc, opt.output_nc, opt.ngf)
netG2.load_state_dict(torch.load(opt.netG2))

###########   Generate   ###########
text_dataset = TestDataset(opt.data_path, opt.size_w, opt.size_h)
train_loader = torch.utils.data.DataLoader(dataset=text_dataset,
                                           batch_size=opt.batch_size,
                                           shuffle=False,
                                           num_workers=6)
loader = iter(train_loader)
number = text_dataset.__len__()

if opt.cuda:
    netG1.cuda()
    netG2.cuda()

print('Number of images: ', number)

for i in range(0, number, opt.batch_size):
    img, name = loader.next()
    print(str(i) + ' ' + name[0])
    flag = False
    if opt.cuda:
        img = img.cuda()
    temp = netG1(img)
    fake = netG2(netG1(img))

    # vutils.save_image(temp.data[0], opt.outf + str(i) + '_mask.png', normalize=True, scale_each=True)
    vutils.save_image(fake.data[0], opt.outf + str(i) + '_image.png', normalize=True, scale_each=True)
