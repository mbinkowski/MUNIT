import torch
import torch.nn as nn
from torch.nn import functional as F
import utils

def get_act(act):
    if act is not None:
        if act == 'tanh':
            return [nn.Tanh()]
        elif act == 'elu':
            return [nn.ELU(alpha=1.0)]
        elif act == 0:
            return [nn.ReLU()]
        elif act > 0:
            return [nn.LeakyReLU(act)]
    return []


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True, sn=False, act=0, bias=False, **kwargs):
    """Custom deconvolutional layer for simplicity."""
    layer = nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=bias)
    layers = [nn.utils.spectral_norm(layer)] if sn else [layer]
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    layers += get_act(act)
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True, sn=False, bias=False, act=0, **kwargs):
    """Custom convolutional layer for simplicity."""
    layer = nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=bias)
    layers = [nn.utils.spectral_norm(layer)] if sn else [layer]
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    layers += get_act(act)
    return nn.Sequential(*layers)

def linear(i_dim, o_dim, bn=True, sn=False, bias=False, act=0, dropout=0, **kwargs):
    layer = nn.Linear(i_dim, o_dim, bias=bias)
    layers = [nn.utils.spectral_norm(layer)] if sn else [layer]
    if dropout > 0:
        layers += [nn.Dropout(dropout)]
    if bn:
        layers.append(nn.BatchNorm1d(o_dim))
    layers += get_act(act)
    return nn.Sequential(*layers)

class Unfold(nn.Module):
    def __init__(self, ks, **kwargs):
        super(Unfold, self).__init__()
        self.f = nn.Unfold(ks, **kwargs)
        self.ks = ks

    def forward(self, x):
        channels = x.size(1)
        x = self.f(x)
        x = x.permute(0, 2, 1).contiguous().view(-1, channels, self.ks, self.ks)
        return x


class ResBlock1(nn.Module):
    def __init__(self, dim=64, bn=False, act=0, bias=True, **kwargs):
        super(ResBlock1, self).__init__()
        self.conv1 = conv(dim, dim, 3, 1, 1, bn=bn, act=act, bias=bias, **kwargs)
        self.conv2 = conv(dim, dim, 3, 1, 1, bn=bn, act=None, bias=bias, **kwargs)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class ResBlockDown(nn.Module):
    def __init__(self, dim=64, bn=False, act=0, bias=True, **kwargs):
        super(ResBlockDown, self).__init__()
        self.conv1 = conv(dim, dim, 3, 2, 1, bn=bn, act=act, bias=bias, **kwargs)
        self.conv2 = conv(dim, dim, 3, 1, 1, bn=bn, act=None, bias=bias, **kwargs)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        skip = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return skip + x


class ResBlockUp(nn.Module):
    def __init__(self, dim=64, bn=False, act=0, bias=True, **kwargs):
        super(ResBlockUp, self).__init__()
        self.conv1 = conv(dim, dim, 3, 1, 1, bn=bn, act=act, bias=bias, **kwargs)
        self.conv2 = deconv(dim, dim, 4, 2, 1, bn=bn, act=None, bias=bias, **kwargs)

    def forward(self, x):
        skip = nn.functional.upsample(x, scale_factor=2, mode='bilinear')
        return self.conv2(self.conv1(x)) + skip


def ResBlocks(n_blocks=2, down_blocks=0, up_blocks=0, return_list=False, **kwargs):
    assert down_blocks * up_blocks == 0
    layers = []
    layers += [ResBlockUp(**kwargs) for _ in range(up_blocks)]
    layers += [ResBlock1(**kwargs) for _ in range(n_blocks)]
    layers += [ResBlockDown(**kwargs) for _ in range(down_blocks)]
    if return_list:
        return layers
    return nn.Sequential(*layers)


class ParConv2d(nn.Module):
    def __init__(self, kernel_size, stride=1, pad=0, dilation=1, **kwargs):
        super(ParConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = (stride, stride)
        self.pad = (pad, pad)
        self.dilation = (dilation, dilation)
        self.mask = None

    def set_mask(self, batch_size):
        mask = torch.ones(batch_size, batch_size, self.kernel_size, self.kernel_size)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    mask[i, j] = 0
        self.mask = utils.cuda(mask)

    def forward(self, x, kernel):
        if self.mask is None:
            self.set_mask(x.size(0))
        _kernel = kernel.repeat(x.size(0), 1, 1, 1) * self.mask
        x = x.permute(1, 0, 2, 3)
        x = F.conv2d(x, _kernel, None, self.stride, self.pad, self.dilation)
        x = x.permute(1, 0, 2, 3)
        return x


class OffsetConv2d(nn.Module):
    def __init__(self, kernel_size, pad=None, **kwargs):
        super(OffsetConv2d, self).__init__()
        if pad is None:
            pad = (kernel_size - 1)//2
        self.kernel_size = kernel_size
        dim = kernel_size**2
        self.linear = nn.Linear(dim, dim, bias=False)
        self.conv = ParConv2d(kernel_size, pad=pad, **kwargs)
#        print('OOFCONV', pad, kernel_size, kwargs)

    def forward(self, x, y):
#        print('OFFSET COVN 2d', x.size(), y.size())
        kernel = self.linear(y)
#        print(kernel.size())
        kernel = F.softmax(kernel, dim=1)
#        print(kernel.size())
        kernel = kernel.view(-1, self.kernel_size, self.kernel_size)
        x = self.conv(x, kernel)
#        print(x.size())
        return x


