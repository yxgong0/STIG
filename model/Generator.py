#Credit: code copied from https://github.com/mrzhu-cool/pix2pix-pytorch/blob/master/models.py
import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.p_relu1 = nn.PReLU()

        self.rconv1 = nn.Conv2d(ngf * 1, ngf * 1, kernel_size=3, stride=1, padding=1)
        self.rconv2 = nn.Conv2d(ngf * 1, ngf * 1, kernel_size=3, stride=1, padding=1)
        self.rconv1_1 = nn.Conv2d(ngf * 1, ngf * 1, kernel_size=3, stride=1, padding=1)
        self.rconv2_1 = nn.Conv2d(ngf * 1, ngf * 1, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(ngf * 1, ngf * 2, kernel_size=4, stride=2, padding=1)
        self.p_relu2 = nn.PReLU()

        self.rconv3 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1)
        self.rconv4 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1)
        self.rconv31 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1)
        self.rconv41 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1)
        self.p_relu3 = nn.PReLU()

        self.rconv5 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1)
        self.rconv6 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1)
        self.rconv51 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1)
        self.rconv61 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.p_relu4 = nn.PReLU()

        self.rconv7 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.rconv8 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.rconv71 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.rconv81 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.p_relu5 = nn.PReLU()

        self.rconv9 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.rconv10 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.rconv91 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.rconv101 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=(4, 6), stride=2, padding=1)

        self.dconv1 = nn.ConvTranspose2d(ngf * 8 * 1, ngf * 8, kernel_size=(4, 6), stride=2, padding=1)

        self.rconv11 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.rconv12 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.rconv111 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.rconv121 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)

        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1)

        self.rconv13 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.rconv14 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.rconv131 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.rconv141 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)

        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1)

        self.rconv15 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1)
        self.rconv16 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1)
        self.rconv151 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1)
        self.rconv161 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1)

        self.dconv4 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1)

        self.rconv17 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1)
        self.rconv18 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1)
        self.rconv171 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1)
        self.rconv181 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1)

        self.dconv5 = nn.ConvTranspose2d(ngf * 2 * 2, ngf * 1, kernel_size=4, stride=2, padding=1)

        self.rconv19 = nn.Conv2d(ngf * 1, ngf * 1, kernel_size=3, stride=1, padding=1)
        self.rconv20 = nn.Conv2d(ngf * 1, ngf * 1, kernel_size=3, stride=1, padding=1)
        self.rconv191 = nn.Conv2d(ngf * 1, ngf * 1, kernel_size=3, stride=1, padding=1)
        self.rconv201 = nn.Conv2d(ngf * 1, ngf * 1, kernel_size=3, stride=1, padding=1)

        self.dconv6 = nn.ConvTranspose2d(ngf * 1 * 2, output_nc, kernel_size=4, stride=2, padding=1)

        self.batch_norm = nn.BatchNorm2d(ngf)
        self.batch_norm2 = nn.BatchNorm2d(ngf * 2)
        self.batch_norm4 = nn.BatchNorm2d(ngf * 4)
        self.batch_norm8 = nn.BatchNorm2d(ngf * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)
        self.p_relu = nn.PReLU()
        self.tanh = nn.Tanh()


    def forward(self, input):
        # Encoder
        # Convolution layers:
        e1 = self.batch_norm(self.conv1(input))
        e1_1 = self.batch_norm(self.rconv1(self.relu(e1)))
        e1_2 = self.batch_norm(self.rconv2(self.relu(e1_1)))
        e1_3 = e1_2 + e1
        e2 = self.batch_norm2(self.conv2(self.p_relu1(e1_3)))
        e2_1 = self.batch_norm2(self.rconv3(self.relu(e2)))
        e2_2 = self.batch_norm2(self.rconv4(self.relu(e2_1)))
        e2_3 = e2_2 + e2
        e3 = self.batch_norm4(self.conv3(self.p_relu2(e2_3)))
        e3_1 = self.batch_norm4(self.rconv5(self.relu(e3)))
        e3_2 = self.batch_norm4(self.rconv6(self.relu(e3_1)))
        e3_3 = e3_2 + e3
        e4 = self.batch_norm8(self.conv4(self.p_relu3(e3_3)))
        e4_1 = self.batch_norm8(self.rconv7(self.relu(e4)))
        e4_2 = self.batch_norm8(self.rconv8(self.relu(e4_1)))
        e4_3 = e4_2 + e4
        e5 = self.batch_norm8(self.conv5(self.p_relu4(e4_3)))
        e5_1 = self.batch_norm8(self.rconv9(self.relu(e5)))
        e5_2 = self.batch_norm8(self.rconv10(self.relu(e5_1)))
        e5_3 = e5_2 + e5
        e6 = self.conv6(self.p_relu5(e5_3))

        # Decoder
        # Deconvolution layers:
        d1_ = self.dropout(self.batch_norm8(self.dconv1(self.relu(e6))))
        d1_1 = self.batch_norm8(self.rconv11(self.relu(d1_)))
        d1_2 = self.batch_norm8(self.rconv12(self.relu(d1_1)))
        d1_3 = d1_2 + d1_
        d1 = torch.cat((d1_3, e5), 1)
        d2_ = self.dropout(self.batch_norm8(self.dconv2(self.relu(d1))))
        d2_1 = self.batch_norm8(self.rconv13(self.relu(d2_)))
        d2_2 = self.batch_norm8(self.rconv14(self.relu(d2_1)))
        d2_3 = d2_2 + d2_
        d2 = torch.cat((d2_3, e4), 1)
        d3_ = self.dropout(self.batch_norm4(self.dconv3(self.relu(d2))))
        d3_1 = self.batch_norm4(self.rconv15(self.relu(d3_)))
        d3_2 = self.batch_norm4(self.rconv16(self.relu(d3_1)))
        d3_3 = d3_2 + d3_
        d3 = torch.cat((d3_3, e3), 1)
        d4_ = self.batch_norm2(self.dconv4(self.relu(d3)))
        d4_1 = self.batch_norm2(self.rconv17(self.relu(d4_)))
        d4_2 = self.batch_norm2(self.rconv18(self.relu(d4_1)))
        d4_3 = d4_2 + d4_
        d4 = torch.cat((d4_3, e2), 1)
        d5_ = self.batch_norm(self.dconv5(self.relu(d4)))
        d5_1 = self.batch_norm(self.rconv19(self.relu(d5_)))
        d5_2 = self.batch_norm(self.rconv20(self.relu(d5_1)))
        d5_3 = d5_2 + d5_
        d5 = torch.cat((d5_3, e1), 1)
        d6 = self.dconv6(self.relu(d5))

        output = self.tanh(d6)

        return output
