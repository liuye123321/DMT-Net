import torch
import torch.nn as nn
from model import common
import torch.nn.functional as F
from resnext.resnext101 import ResNeXt101
from model.common import UNet
def make_model():
    return DRND_v4()


class DRND_v4(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(DRND_v4, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225
        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        self.phase = 2
        self.n_feats = 32
        self.n_colors = 3
        kernel_size = 3

        act = nn.ReLU(True)
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down0 = nn.Sequential(
            nn.Conv2d(64, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(64, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(256, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down7 = nn.Sequential(
            nn.Conv2d(512, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down8 = nn.Sequential(
            nn.Conv2d(1024, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down9 = nn.Sequential(
            nn.Conv2d(2048, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down10 = nn.Sequential(
            nn.Conv2d(64, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down11 = nn.Sequential(
            nn.Conv2d(256, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down12 = nn.Sequential(
            nn.Conv2d(512, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down13 = nn.Sequential(
            nn.Conv2d(1024, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.down14 = nn.Sequential(
            nn.Conv2d(2048, self.n_feats, kernel_size=1), nn.SELU()
        )
        self.decoder_A1 = common.decoder(conv, self.n_feats, kernel_size, 2, act)  #16
        self.decoder_A2 = common.decoder(conv, self.n_feats, kernel_size, 2, act)  # 16
        self.decoder_A3 = common.decoder(conv, self.n_feats, kernel_size, 2, act)  # 16
        self.decoder_A4 = common.decoder(conv, self.n_feats, kernel_size, 2, act)  # 16


        self.decoder_T1 = common.decoder(conv, self.n_feats, kernel_size, 20, act)
        self.decoder_T2 = common.decoder(conv, self.n_feats, kernel_size, 20, act)
        self.decoder_T3 = common.decoder(conv, self.n_feats, kernel_size, 20, act)
        self.decoder_T4 = common.decoder(conv, self.n_feats, kernel_size, 20, act)


        self.decoder_J1 = common.decoder(conv, self.n_feats, kernel_size, 20, act)
        self.decoder_J2 = common.decoder(conv, self.n_feats, kernel_size, 20, act)
        self.decoder_J3 = common.decoder(conv, self.n_feats, kernel_size, 20, act)
        self.decoder_J4 = common.decoder(conv, self.n_feats, kernel_size, 20, act)

        self.conv1 = nn.Sequential(conv(self.n_feats * 2, self.n_feats, 1),
                                   nn.PReLU()
                                   )
        self.conv2 = nn.Sequential(conv(self.n_feats * 2, self.n_feats, 1),
                                   nn.PReLU()
                                   )
        self.conv3 = nn.Sequential(conv(self.n_feats * 2, self.n_feats, 1),
                                   nn.PReLU()
                                   )
        self.conv4 = nn.Sequential(conv(self.n_feats * 2, self.n_feats, 1),
                                   nn.PReLU()
                                   )
        self.conv5 = nn.Sequential(conv(self.n_feats * 2, self.n_feats, 1),
                                   nn.PReLU()
                                   )
        self.conv6 = nn.Sequential(conv(self.n_feats * 2, self.n_feats, 1),
                                   nn.PReLU()
                                   )
        self.conv7 = nn.Sequential(conv(self.n_feats * 2, self.n_feats, 1),
                                   nn.PReLU()
                                   )
        self.conv8 = nn.Sequential(conv(self.n_feats * 2, self.n_feats, 1),
                                   nn.PReLU()
                                   )
        self.conv9 = nn.Sequential(conv(self.n_feats * 2, self.n_feats, 1),
                                   nn.PReLU()
                                   )
        self.conv10 = nn.Sequential(conv(self.n_feats * 2, self.n_feats, 1),
                                   nn.PReLU()
                                   )
        self.conv11 = nn.Sequential(conv(self.n_feats * 2, self.n_feats, 1),
                                   nn.PReLU()
                                   )
        self.conv12 = nn.Sequential(conv(self.n_feats * 2, self.n_feats, 1),
                                   nn.PReLU()
                                   )

        self.tail_T4 = conv(self.n_feats, self.n_colors, kernel_size)
        self.tail_J4 = conv(self.n_feats, self.n_colors, kernel_size)
        self.tail_A4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, self.n_feats, kernel_size=1), nn.SELU(),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=1), nn.SELU(), nn.Dropout(0.2),
            nn.Conv2d(self.n_feats, 1, kernel_size=1), nn.Sigmoid()
        )

        self.refineA = UNet(1, 1, 8)
        self.refineT = UNet(self.n_colors, self.n_colors, 16)
        self.refineJ = UNet(self.n_colors, self.n_colors, 32)

    def forward(self, input):
        x = (input - self.mean) / self.std
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down0 = self.down0(layer0)  # 64,256,256
        down1 = self.down1(layer1)  # 64,128,128
        down2 = self.down2(layer2)  # 64,64,64
        down3 = self.down3(layer3)  # 64,32,32
        down4 = self.down4(layer4)  # 64,16,16

        down5 = self.down5(layer0)  # 64,256,256
        down6 = self.down6(layer1)  # 64,128,128
        down7 = self.down7(layer2)  # 64,64,64
        down8 = self.down8(layer3)  # 64,32,32
        down9 = self.down9(layer4)  # 64,32,32

        down10 = self.down10(layer0)  # 64,256,256
        down11 = self.down11(layer1)  # 64,128,128
        down12 = self.down12(layer2)  # 64,64,64
        down13 = self.down13(layer3)  # 64,32,32
        down14 = self.down14(layer4)  # 64,32,32
        # upsample to SR features
        FT1 = self.decoder_T1(down4)
        FJ1 = self.decoder_J1(down9)
        FA1 = self.decoder_A1(down14)
        uT1 = torch.cat((FT1, down3), 1)
        uJ1 = torch.cat((FJ1, down8), 1)
        uA1 = torch.cat((FA1, down13), 1)

        uT1 = self.conv1(uT1)
        uJ1 = self.conv2(uJ1)
        uA1 = self.conv3(uA1)

        FT2 = self.decoder_T2(uT1)
        FJ2 = self.decoder_J2(uJ1)
        FA2 = self.decoder_A2(uA1)
        uT2 = torch.cat((FT2, down2), 1)
        uJ2 = torch.cat((FJ2, down7), 1)
        uA2 = torch.cat((FA2, down12), 1)
        uT2 = self.conv4(uT2)
        uJ2 = self.conv5(uJ2)
        uA2 = self.conv6(uA2)

        FT3 = self.decoder_T3(uT2)
        FJ3 = self.decoder_J3(uJ2)
        FA3 = self.decoder_A3(uA2)
        uT3 = torch.cat((FT3, down1), 1)
        uJ3 = torch.cat((FJ3, down6), 1)
        uA3 = torch.cat((FA3, down11), 1)
        uT3 = self.conv7(uT3)
        uJ3 = self.conv8(uJ3)
        uA3 = self.conv9(uA3)

        FT4 = self.decoder_T4(uT3)
        FJ4 = self.decoder_J4(uJ3)
        FA4 = self.decoder_A4(uA3)
        uT4 = torch.cat((FT4, down0), 1)
        uJ4 = torch.cat((FJ4, down5), 1)
        uA4 = torch.cat((FA4, down10), 1)
        uT4 = self.conv10(uT4)
        uJ4 = self.conv11(uJ4)
        uA4 = self.conv12(uA4)
        T = self.tail_T4(uT4)
        J = self.tail_J4(uJ4)
        A = self.tail_A4(uA4)
        J = (J * self.std + self.mean).clamp(min=0, max=1)
        haze = J * T + A * (1 - T)
        A2 = A + self.refineA(A)
        T2 = T + self.refineT(T)
        J2 = J + self.refineJ(J)
        haze2 = J2 * T2 + A * (1 - T2)
        predict1 = [J, T, A, haze]
        predict2 = [J2, T2, A2, haze2]
        return predict1, predict2

