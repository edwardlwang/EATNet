import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

from net.pvtv2 import pvt_v2


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class EEM(nn.Module):
    def __init__(self):
        super(EEM, self).__init__()
        # -------------Upsampling--------------#
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        # -------------Poolling--------------#
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.pool4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.pool8 = nn.MaxPool2d(8, 8, ceil_mode=True)
        # -------------Decoder Label--------------#
        # stage 4
        self.decoder_l4 = nn.Sequential(
            nn.Conv2d(512, 1, 3, dilation=2, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        # stage3
        self.decoder_l3 = nn.Sequential(
            nn.Conv2d(322, 1, 3, dilation=2, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        # stage2
        self.decoder_l2 = nn.Sequential(
            nn.Conv2d(130, 1, 3, dilation=2, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        # stage1
        self.decoder_l1 = nn.Sequential(
            nn.Conv2d(66, 1, 3, dilation=2, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        # -------------Decoder Edge--------------#
        # stage 4
        self.decoder_e4 = nn.Sequential(
            nn.Conv2d(512, 1, 3, dilation=2, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        # stage 3
        self.decoder_e3 = nn.Sequential(
            nn.Conv2d(833, 1, 3, dilation=2, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        # stage 2
        self.decoder_e2 = nn.Sequential(
            nn.Conv2d(449, 1, 3, dilation=2, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        # stage 1
        self.decoder_e1 = nn.Sequential(
            nn.Conv2d(193, 1, 3, dilation=2, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        # -------------Side output--------------#
        ## Edge
        self.conv_oute1 = nn.Conv2d(1, 1, 3, padding=1)
        ## Sal
        self.conv_out = nn.Conv2d(4, 1, 3, padding=1)
        # -------------EPAU--------------#
        # 注：原先还存在 conv_oute2/3/4 与 conv_epau5，但它们在 forward 中从未被调用，
        # 在完整的四路损失下也收不到梯度，共 1.25M 死参数，已删除。
        # 因此 2026-09-22 之前训练的 checkpoint 会多出这几个键（加载时忽略即可）。
        self.conv_epau4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, padding=0),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.conv_epau3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.conv_epau2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, padding=0),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.conv_epau1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, padding=0),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4):
        # stage 4
        xe4 = self.decoder_e4(x4)
        t = torch.cat((xe4, x4), 1)
        t = self.upscore2(t)

        # stage 3
        xe3 = self.decoder_e3(torch.cat((t, x3), 1))
        t = torch.cat((xe3, x3), 1)
        t = self.upscore2(t)

        # stage 2
        xe2 = self.decoder_e2(torch.cat((t, x2), 1))
        t = torch.cat((xe2, x2), 1)
        t = self.upscore2(t)

        # stage 1
        xe1 = self.decoder_e1(torch.cat((t, x1), 1))
        oute1 = self.conv_oute1(xe1)
        oute1 = torch.sigmoid(oute1)  # turn to 0-1
        # -------------Copy SE to different channels--------------#
        SE = torch.cat((oute1, oute1), 1)
        SE = torch.cat((SE, SE), 1)
        SE = torch.cat((SE, SE), 1)
        SE = torch.cat((SE, SE), 1)
        SE = torch.cat((SE, SE), 1)
        SE_64 = torch.cat((SE, SE), 1)
        SE_128 = torch.cat((SE_64, SE_64), 1)
        SE_256 = torch.cat((SE_128, SE_128), 1)
        # -------------Decoder Label--------------#
        # stage 5
        xl4 = self.decoder_l4(x4)
        # #EPAU
        SE_down4 = self.pool8(SE_256)
        OE_4 = torch.mul(SE_down4, xl4) + xl4
        PA4 = self.conv_epau4(OE_4)
        t = torch.cat((xl4, PA4), 1)
        t = self.upscore2(t)

        # stage 3
        xl3 = self.decoder_l3(torch.cat((t, x3), 1))
        ##EPAU
        SE_down3 = self.pool4(SE_128)
        OE_3 = torch.mul(SE_down3, xl3) + xl3
        PA3 = self.conv_epau3(OE_3)
        t = torch.cat((xl3, PA3), 1)
        t = self.upscore2(t)

        # stage 2
        xl2 = self.decoder_l2(torch.cat((t, x2), 1))
        ##EPAU
        SE_down2 = self.pool2(SE_64)
        OE_2 = torch.mul(SE_down2, xl2) + xl2
        PA2 = self.conv_epau2(OE_2)
        t = torch.cat((xl2, PA2), 1)
        t = self.upscore2(t)

        # stage 1
        xl1 = self.decoder_l1(torch.cat((t, x1), 1))
        ##EPAU
        OE_1 = torch.mul(SE_64, xl1) + xl1
        PA1 = self.conv_epau1(OE_1)
        # -------------Side Output--------------#
        outl1 = PA1
        outl2 = self.upscore2(PA2)
        outl3 = self.upscore4(PA3)
        outl4 = self.upscore8(PA4)
        out = torch.cat((outl1, outl2, outl3, outl4), 1)
        out = self.conv_out(out)

        return out


class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU()
        )


class RFEM(nn.Module):
    def __init__(self, channels):
        super(RFEM, self).__init__()

        self.conv1_1x1 = nn.Conv2d(3 * channels // 2, channels // 4, 1)
        self.conv2_1x1 = nn.Conv2d(7 * channels // 4, channels // 4, 1)
        self.conv3_1x1 = nn.Conv2d(2 * channels, channels, 1)
        self.dconv_r2 = ConvBNReLU(in_channels=channels, out_channels=channels // 2, kernel_size=3, dilation=2)
        self.dconv_r4 = ConvBNReLU(in_channels=channels // 4, out_channels=channels // 4, kernel_size=3, dilation=4)
        self.dconv_r8 = ConvBNReLU(in_channels=channels // 4, out_channels=channels // 4, kernel_size=3, dilation=8)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        r1 = self.dconv_r2(x)
        r1 = torch.cat([r1, x], dim=1)

        r2 = self.dconv_r4(self.conv1_1x1(r1))
        r2 = torch.cat([r2, r1], dim=1)

        r3 = self.dconv_r8(self.conv2_1x1(r2))
        r3 = torch.cat([r3, r2], dim=1)

        out = self.conv3_1x1(r3)

        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pvt = pvt_v2(pretrained=True)
        self.eem = EEM()

        self.efm1 = EFM(64)
        self.efm2 = EFM(128)
        self.efm3 = EFM(320)
        self.efm4 = EFM(512)

        self.reduce1 = Conv1x1(64, 64)
        self.reduce2 = Conv1x1(128, 128)
        self.reduce3 = Conv1x1(320, 256)
        self.reduce4 = Conv1x1(512, 256)

        self.RFEM1 = RFEM(704)
        self.RFEM2 = RFEM(640)
        self.RFEM3 = RFEM(512)

        self.predictor1 = nn.Conv2d(704, 1, 1)
        self.predictor2 = nn.Conv2d(640, 1, 1)
        self.predictor3 = nn.Conv2d(512, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.pvt(x)

        edge = self.eem(x1, x2, x3, x4)
        edge_att = torch.sigmoid(edge)

        x1a = self.efm1(x1, edge_att)
        x2a = self.efm2(x2, edge_att)
        x3a = self.efm3(x3, edge_att)
        x4a = self.efm4(x4, edge_att)

        x1r = self.reduce1(x1a)
        x2r = self.reduce2(x2a)
        x3r = self.reduce3(x3a)
        x4r = self.reduce4(x4a)

        x34 = self.RFEM3(x3r, x4r)
        x234 = self.RFEM2(x2r, x34)
        x1234 = self.RFEM1(x1r, x234)

        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        return o3, o2, o1, oe
