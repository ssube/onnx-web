# from https://github.com/cszn/BSRGAN/blob/main/models/network_rrdbnet.py

from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

logger = getLogger(__name__)


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers, **kwarg):
    layers = []
    for _ in range(n_layers):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNetRescale(nn.Module):
    """
    RRDBNet with variable input channels based on scale.
    This is the format expected by the official models.
    In this architecture, the modules stay the same but input channels change.
    """

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    ):
        super(RRDBNetRescale, self).__init__()
        self.scale = scale

        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        logger.trace(
            "RRDBNetRescale params: %s",
            [num_in_ch, num_out_ch, num_feat, num_block, num_grow_ch, scale],
        )

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.body = make_layer(RRDB, num_block, nf=num_feat, gc=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        trunk = self.conv_body(self.body(feat))
        feat = feat + trunk

        if self.scale > 1:
            feat = self.lrelu(
                self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
            )

        if self.scale == 4:
            feat = self.lrelu(
                self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
            )

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        return out


class RRDBNetFixed(nn.Module):
    """
    RRDBNet with fixed input channels regardless of scale.
    This is the format expected by many third-party models.
    In this architecture, the modules come and go based on scale, but the input channels stay the same.
    """

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    ):
        super(RRDBNetFixed, self).__init__()
        self.scale = scale

        logger.trace(
            "RRDBNetFixed params: %s",
            [num_in_ch, num_out_ch, num_feat, num_block, num_grow_ch, scale],
        )

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.body = make_layer(RRDB, num_block, nf=num_feat, gc=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # upsampling
        if self.scale > 1:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.conv_body(self.body(feat))
        feat = feat + trunk

        if self.scale > 1:
            feat = self.lrelu(
                self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
            )
            feat = self.lrelu(
                self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
            )

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        return out
