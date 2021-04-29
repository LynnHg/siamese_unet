import torch
import math
from torch import nn
import torch.nn.functional as F
from networks.custom_modules.basic_modules import *

class Siamese_UNet(nn.Module):
    def __init__(self, img_ch=1, num_classes=3, aux_classes=1, depth=3, use_deconv=False):
        super(Baseline, self).__init__()
        chs = [36, 72, 144, 288, 360]
        self.pool = nn.MaxPool2d(2, 2)

        self.p1_enc1 = EncoderBlock(img_ch, chs[0], depth=depth)
        self.p1_enc2 = EncoderBlock(chs[0], chs[1], depth=depth)
        self.p1_enc3 = EncoderBlock(chs[1], chs[2], depth=depth)
        self.p1_enc4 = EncoderBlock(chs[2], chs[3], depth=depth)
        self.p1_cen = EncoderBlock(chs[3], chs[4], depth=depth)

        self.p1_dec4 = DecoderBlock(chs[4] * 2, chs[3], use_deconv=use_deconv)
        self.p1_decconv4 = EncoderBlock(chs[3] * 3, chs[3])

        self.p1_dec3 = DecoderBlock(chs[3], chs[2], use_deconv=use_deconv)
        self.p1_decconv3 = EncoderBlock(chs[2] * 3, chs[2])

        self.p1_dec2 = DecoderBlock(chs[2], chs[1], use_deconv=use_deconv)
        self.p1_decconv2 = EncoderBlock(chs[1] * 3, chs[1])

        self.p1_dec1 = DecoderBlock(chs[1], chs[0], use_deconv=use_deconv)
        self.p1_decconv1 = EncoderBlock(chs[0] * 3, chs[0])

        self.p1_conv_1x1 = nn.Conv2d(chs[0], num_classes, kernel_size=1, stride=1, padding=0)

        self.p2_enc1 = EncoderBlock(img_ch, chs[0], depth=depth)
        self.p2_enc2 = EncoderBlock(chs[0], chs[1], depth=depth)
        self.p2_enc3 = EncoderBlock(chs[1], chs[2], depth=depth)
        self.p2_enc4 = EncoderBlock(chs[2], chs[3], depth=depth)
        self.p2_cen = EncoderBlock(chs[3], chs[4], depth=depth)

        self.p2_dec4 = DecoderBlock(chs[4], chs[3])
        self.p2_decconv4 = EncoderBlock(chs[3] * 2, chs[3])

        self.p2_dec3 = DecoderBlock(chs[3], chs[2])
        self.p2_decconv3 = EncoderBlock(chs[2] * 2, chs[2])

        self.p2_dec2 = DecoderBlock(chs[2], chs[1])
        self.p2_decconv2 = EncoderBlock(chs[1] * 2, chs[1])

        self.p2_dec1 = DecoderBlock(chs[1], chs[0])
        self.p2_decconv1 = EncoderBlock(chs[0] * 2, chs[0])

        self.p2_conv_1x1 = nn.Conv2d(chs[0], aux_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # p1 encoder
        p1_x1 = self.p1_enc1(x)
        p1_x2 = self.pool(p1_x1)
        p1_x2 = self.p1_enc2(p1_x2)
        p1_x3 = self.pool(p1_x2)
        p1_x3 = self.p1_enc3(p1_x3)
        p1_x4 = self.pool(p1_x3)
        p1_x4 = self.p1_enc4(p1_x4)
        p1_center = self.pool(p1_x4)
        p1_center = self.p1_cen(p1_center)

        p2_x1 = self.p2_enc1(x)
        p2_x2 = self.pool(p2_x1)
        p2_x2 = self.p2_enc2(p2_x2)
        p2_x3 = self.pool(p2_x2)
        p2_x3 = self.p2_enc3(p2_x3)
        p2_x4 = self.pool(p2_x3)
        p2_x4 = self.p2_enc4(p2_x4)
        p2_center = self.pool(p2_x4)
        p2_center = self.p2_cen(p2_center)

        p2_d4 = self.p2_dec4(p2_center)
        p2_d4 = torch.cat((p2_x4, p2_d4), dim=1)
        p2_d4 = self.p2_decconv4(p2_d4)

        p2_d3 = self.p2_dec3(p2_d4)
        p2_d3 = torch.cat((p2_x3, p2_d3), dim=1)
        p2_d3 = self.p2_decconv3(p2_d3)

        p2_d2 = self.p2_dec2(p2_d3)
        p2_d2 = torch.cat((p2_x2, p2_d2), dim=1)
        p2_d2 = self.p2_decconv2(p2_d2)

        p2_d1 = self.p2_dec1(p2_d2)
        p2_d1 = torch.cat((p2_x1, p2_d1), dim=1)
        p2_d1 = self.p2_decconv1(p2_d1)

        p2_out = self.p2_conv_1x1(p2_d1)
   
        d4 = self.p1_dec4(torch.cat([p1_center, p2_center], dim=1))
        d4 = torch.cat((p1_x4, d4, p2_x4), dim=1)
        d4 = self.p1_decconv4(d4)

        d3 = self.p1_dec3(d4)
        d3 = torch.cat((p1_x3, d3, p2_x3), dim=1)
        d3 = self.p1_decconv3(d3)

        d2 = self.p1_dec2(d3)
        d2 = torch.cat((p1_x2, d2, p2_x2), dim=1)
        d2 = self.p1_decconv2(d2)

        d1 = self.p1_dec1(d2)
        d1 = torch.cat((p1_x1, d1, p2_x1), dim=1)
        d1 = self.p1_decconv1(d1)

        p1_out = self.p1_conv_1x1(d1)

        return p1_out, p2_out
