import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
from residual import ResidualBlock
from torchvision.models.resnet import BasicBlock, ResNet
from layers import *

class ID(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ID, self).__init__()

    def forward(self, x, *args):
        return x

NoNoise = ID


class Noise(nn.Module):
    def __init__(self, size, std=1.):
        super(Noise, self).__init__()
        self.noise = torch.autograd.Variable(utils.get_ones(size))
        self.std = std

    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise.view(x.size())

class ConcatNoise(Noise):
    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        return torch.cat((x, self.noise), dim=1)


class ConcatNoise1(nn.Module):
    def __init__(self, size=32, z_dim=64, std=1, return_vector=False):
        super(ConcatNoise1, self).__init__()
        self.return_vector = return_vector
        self.std = std
        self.size = size
        self.noise = torch.autograd.Variable(utils.get_ones((size[0], z_dim)))
        self.fc = linear(z_dim, int(np.prod(size[1:])), bias=True)

    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        z = self.fc(self.noise)
        r = torch.cat((x, z.view(self.size)), dim=1)
        if self.return_vector:
            return r, self.noise
        return r


class ConcatNoise2(nn.Module):
    def __init__(self, noise_dim, size=32, std=1, act=.2, sn=False, bn=False, **kwargs):
        super(ConcatNoise2, self).__init__()
        self.std = std
        self.size = size = size//8
        self.noise_dim = noise_dim
        self.fc = linear(noise_dim, 2*noise_dim*size**2, bias=True, sn=sn, bn=bn, act=act)
        self.deconv1 = deconv(2*noise_dim, noise_dim, 4, stride=2, pad=1, act=act, sn=sn, bn=False)

    def forward(self, x, z):
        z = self.fc(z)
        z = z.view(-1, 2*self.noise_dim, self.size, self.size)
        z = self.deconv1(z)
        return torch.cat((x, z), dim=1)


class ConcatNoise3_2(nn.Module):
    downsample_factor = 2
    def __init__(self, noise_dim, size=32, std=1, act=.2, sn=False,
                 top_sn=False, bn=False, **kwargs):
        super(ConcatNoise3_2, self).__init__()
        self.std = std
        self.size = size = size//8
        self.noise_dim = noise_dim
        self.fc = linear(noise_dim, 4*noise_dim*size**2, bias=True, sn=False, bn=bn, act=act)
        self.deconv1 = deconv(4*noise_dim, 2*noise_dim, 4, stride=2, pad=1, act=act, sn=sn, bn=bn)
        self.deconv2 = deconv(2*noise_dim, noise_dim, 4, stride=2, pad=1, act=act,
                              sn=top_sn, bn=False)

    def forward(self, x, z):
        z = self.fc(z)
        z = z.view(-1, 4*self.noise_dim, self.size, self.size)
        z = self.deconv1(z)
        z = self.deconv2(z)
        return torch.cat((x, z), dim=1)

class Noise2Channels_2(nn.Module):
    downsample_factor = 2
    def __init__(self, size=32, **kwargs):
        super(Noise2Channels_2, self).__init__()
        self.s = size // self.downsample_factor

    def forward(self, x, z):
        z0, z1 = z.size(0), z.size(1)
        z = z.repeat(self.s, self.s, 1, 1).permute(2,3,0,1).view(z0, z1, self.s, self.s)
        return torch.cat((x, z), dim=1)


class Noise2Channels_4(Noise2Channels_2):
    downsample_factor = 4


class Noise2Channels_8(Noise2Channels_2):
    downsample_factor = 8


class ConcatOffsetNoise3_2(nn.Module):
    def __init__(self, o_dim, offset_ks, noise_dim=64, **kwargs):
        assert noise_dim > offset_ks**2, 'noise dimension must be no less than offset kernel'
        super(ConcatOffsetNoise3_2, self).__init__()
        self.offset_ks = offset_ks
        self.concat = ConcatNoise3_2(o_dim, noise_dim=noise_dim - offset_ks**2, **kwargs)
        self.offset = OffsetConv2d(offset_ks, padding=(offset_ks-1)//2)

    def forward(self, x, z):
        n_off = self.offset_ks**2
        x = self.offset(x, z[:, :n_off])
        return self.concat(x, z[:, n_off:])

class ResNet4_2(nn.Module):
    def __init__(self, i_dim=3, o_dim=1, conv_dim=64, size=32, act=0, **kwargs):
        super(ResNet4_2, self).__init__()
        self.res1 = ResidualBlock(i_dim, conv_dim, None, size, act=act)
        self.res2 = ResidualBlock(conv_dim, conv_dim, None, size, act=act)
        self.res3 = ResidualBlock(conv_dim, 2*conv_dim, 'down', size, act=act)
        self.res4 = ResidualBlock(2*conv_dim, o_dim, None, size//2, act=act)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return x


class ResNet4x2(ResNet4_2):
    def __init__(self, i_dim=1, o_dim=3, conv_dim=128, size=16, act=0,
                 top_act='same', **kwargs):
        super(ResNet4x2, self).__init__()
        self.res1 = ResidualBlock(i_dim, 2*conv_dim, None, size, act=act)
        self.res2 = ResidualBlock(2*conv_dim, conv_dim, 'up', size, act=act)
        self.res3 = ResidualBlock(conv_dim, conv_dim, None, size*2, act=act)
        self.res4 = ResidualBlock(conv_dim, o_dim, None, size*2, act=act,
                                  top_act=top_act)


class ResNet2x2(nn.Module):
    def __init__(self, i_dim=1, o_dim=3, conv_dim=128, size=16, act=0,
                 top_act='same', **kwargs):
        super(ResNet2x2, self).__init__()
        self.res1 = ResidualBlock(i_dim, conv_dim, None, size, act=act)
        self.res2 = ResidualBlock(conv_dim, o_dim, 'up', size, act=act,
                                  top_act=top_act)

    def forward(self, x):
        x = self.res1(x)
        return self.res2(x)


class ResNet6_8(nn.Module):
    def __init__(self, i_dim=1, o_dim=1, conv_dim=64, size=32, act=0,
                 top_activation=False, activation_penalty=False, **kwargs):
        super(ResNet6_8, self).__init__()
        self.activation_penalty = activation_penalty
        self.res1 = ResidualBlock(i_dim, conv_dim, 'down', size, act=act)
        self.res2 = ResidualBlock(conv_dim, conv_dim, None, size//2, act=act)
        self.res3 = ResidualBlock(conv_dim, 2*conv_dim, 'down', size//2, act=act)
        self.res4 = ResidualBlock(2*conv_dim, 2*conv_dim, None, size//4, act=act)
        self.res5 = ResidualBlock(2*conv_dim, 4*conv_dim, 'down', size//4, act=act)
        self.res6 = ResidualBlock(4*conv_dim, 4*conv_dim, None, size//8, act=act)
        self.top = linear(((size//8)**2) * 4 * conv_dim, o_dim,
                          act=act if top_activation else None, bn=False)

    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.res5(x4)
        x6 = self.res6(x5)
        x6 = x6.view(x6.size(0), -1)
        x7 = self.top(x6)
        if self.activation_penalty:
            return x7, utils.activation_penalty([x1, x2, x3, x4, x5, x6, x7])
        return x7


class ResNet11(nn.Module):
    def __init__(self, i_dim, o_dim, conv_dim, size=32, act=0, top_activation=False, **kwargs):
        super(ResNet11, self).__init__()
        self.downsample2 = ResNet4_2(i_dim, 2*conv_dim, conv_dim, size, act)
        self.top = linear((size//2)**2 * 2 * conv_dim, o_dim,
                          act=act if top_activation else None, bn=False)

    def forward(self, x):
        x = self.downsample2(x)
        x = x.view(x.size(0), -1)
        x = self.top(x)
        return x


class ResNet16(ResNet):
    def __init__(self, block=BasicBlock, i_dim=3, o_dim=1, conv_dim=64, size=32, act=None, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = i_dim
        self.layer1 = self._make_layer(block, conv_dim, 2)
        self.layer2 = self._make_layer(block, 2*conv_dim, 2, stride=2)
        self.layer3 = self._make_layer(block, 4*conv_dim, 2, stride=2)
        self.avgpool = nn.AvgPool2d(size//4, stride=1)
        self.fc = linear(4*conv_dim * block.expansion, o_dim, bn=False, act=act)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class InvConvNet5(nn.Module):
    """Decoder"""
    def __init__(self, conv_dim=64, i_dim=1024, o_dim=1, size=32, act=0, sn=False, **kwargs):
        super(InvConvNet5, self).__init__()
        self.conv_dim = conv_dim
        self.size0 = size//16
        self.l = linear(i_dim, conv_dim*8*(self.size0)**2, bias=True, sn=sn)
        self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4, act=act, sn=sn)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4, act=act, sn=sn)
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4, act=act, sn=sn)
        self.deconv4 = deconv(conv_dim, o_dim, 4, bn=False, act='tanh', sn=sn)
    
    def forward(self, x):
        x = self.l(x)          # (?, 64*8*2*2)
        x = x.view(x.size(0), self.conv_dim*8, self.size0, self.size0) # (?, 64*8, 2, 2) 
        x = self.deconv1(x)  # (?, 256, 4, 4)
        x = self.deconv2(x)  # (?, 128, 8, 8)
        x = self.deconv3(x)  # (?, 256, 16, 16)
        x = self.deconv4(x)  # (?, 3, 32, 32)
        return x

class FullyInvConvNet2(nn.Module):
    """Decoder"""
    def __init__(self, conv_dim=64, i_dim=1024, o_dim=1, size=32, bn=True, act=0, sn=False, **kwargs):
        super(FullyInvConvNet2, self).__init__()
        self.deconv1 = deconv(i_dim, conv_dim, 4, bn=bn, act=act, sn=sn)
        self.deconv2 = deconv(conv_dim, o_dim, 4, bn=False, act='tanh', sn=sn)
    
    def forward(self, x):
        x = self.deconv1(x)  # (?, 128, 8, 8)
        x = self.deconv2(x)  # (?, 256, 16, 16)
        return x

class FullyInvConvNet2_2(nn.Module):
    """Decoder"""
    def __init__(self, conv_dim=64, i_dim=1024, o_dim=1, size=32, bn=True, act=0, sn=False, **kwargs):
        super(FullyInvConvNet2_2, self).__init__()
        self.deconv1 = deconv(i_dim, conv_dim, 1, stride=1, pad=0, bn=bn, act=act, sn=sn)
        self.deconv2 = deconv(conv_dim, o_dim, 4, bn=False, act='tanh', sn=sn)
    
    def forward(self, x):
        x = self.deconv1(x)  # (?, 128, 8, 8)
        x = self.deconv2(x)  # (?, 256, 16, 16)
        return x


class FullyInvConvNet3_2(nn.Module):
    """Decoder"""
    def __init__(self, conv_dim=64, i_dim=1024, o_dim=1, size=32, bn=True, act=0,
                 sn=False, top_sn=False, res_blocks=0, **kwargs):
        super(FullyInvConvNet3_2, self).__init__()
        self.deconv1 = deconv(i_dim, conv_dim, 1, stride=1, pad=0, bn=bn, act=act, sn=sn)

        if res_blocks > 0:
            self.mid = ResBlocks(n_blocks=res_blocks, dim=conv_dim, bn=bn, act=act)
            self.deconv3 = deconv(conv_dim, o_dim, 4, bn=False, act='tanh', sn=top_sn)
        else:
            self.mid = deconv(conv_dim, conv_dim, 4, bn=bn, act=act, sn=sn)
            self.deconv3 = deconv(conv_dim, o_dim, 1, stride=1, pad=0, bn=False, act='tanh', sn=top_sn)
    
    def forward(self, x):
        x = self.deconv1(x)  # (?, 128, 8, 8)
        x = self.mid(x)  # (?, 256, 16, 16)
        x = self.deconv3(x)  # (?, 3, 32, 32)
        return x


class FullyInvConvNet3_4(FullyInvConvNet3_2):
    """Decoder"""
    def __init__(self, conv_dim=64, i_dim=1024, o_dim=1, size=32, bn=True, act=0,
                 sn=False, top_sn=False, res_blocks=0, **kwargs):
        super(FullyInvConvNet3_4, self).__init__()
        self.deconv1 = deconv(i_dim, conv_dim, 4, bn=bn, act=act, sn=sn)

        if res_blocks > 0:
            self.mid = ResBlocks(n_blocks=res_blocks, dim=conv_dim, bn=bn, act=act)
        else:
            self.mid = deconv(conv_dim, conv_dim, 1, stride=1, pad=0, bn=bn, act=act, sn=sn)

        self.deconv3 = deconv(conv_dim, o_dim, 4, bn=False, act='tanh', sn=top_sn)


class FullyInvConvNet3_8(FullyInvConvNet3_2):
    """Decoder"""
    def __init__(self, conv_dim=64, i_dim=1024, o_dim=1, size=32, bn=True, act=0,
                 sn=False, top_sn=False, res_blocks=0, **kwargs):
        super(FullyInvConvNet3_8, self).__init__()
        self.deconv1 = deconv(i_dim, conv_dim, 4, bn=bn, act=act, sn=sn)

        if res_blocks > 0:
            self.mid = ResBlocks(n_blocks=res_blocks - 1, up_blocks=1, dim=conv_dim, bn=bn, act=act)
            self.deconv3 = deconv(conv_dim, o_dim, 4, bn=False, act='tanh', sn=top_sn)
        else:
            self.mid = deconv(conv_dim, conv_dim//2, 4, bn=bn, act=act, sn=sn)
            self.deconv3 = deconv(conv_dim//2, o_dim, 4, bn=False, act='tanh', sn=top_sn)


class FullyInvConvNet4(nn.Module):
    """Decoder"""
    def __init__(self, conv_dim=64, i_dim=1024, o_dim=1, size=32, bn=True, mid_ks=1, act=0, sn=False, **kwargs):
        super(FullyInvConvNet4, self).__init__()
        self.deconv1 = deconv(i_dim, 4*conv_dim, 4, bn=bn, act=act, sn=sn)
        self.deconv2 = deconv(4*conv_dim, 2*conv_dim, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=bn, act=act, sn=sn)
        self.deconv3 = deconv(2*conv_dim, conv_dim, 4, bn=bn, act=act, sn=sn)
        self.deconv4 = deconv(conv_dim, o_dim, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=False, act='tanh', sn=sn)
    
    def forward(self, x):
        x = self.deconv1(x)  # (?, 512, 16, 16)
        x = self.deconv2(x)  # (?, 256, 16, 16)
        x = self.deconv3(x)  # (?, 128, 32, 32)
        x = self.deconv4(x)  # (?, 3, 32, 32)
        return x

class FullyInvConvNet4_4(nn.Module):
    """Decoder"""
    def __init__(self, conv_dim=64, i_dim=1024, o_dim=1, size=32, bn=True, act=0, sn=False, **kwargs):
        super(FullyInvConvNet4_4, self).__init__()
        self.deconv1 = deconv(i_dim, 4*conv_dim, 1, stride=1, pad=0, bn=bn, act=act, sn=sn)
        self.deconv2 = deconv(4*conv_dim, 2*conv_dim, 4, bn=bn, act=act, sn=sn)
        self.deconv3 = deconv(2*conv_dim, conv_dim, 1, stride=1, pad=0, bn=bn, act=act, sn=sn)
        self.deconv4 = deconv(conv_dim, o_dim, 4, bn=False, act='tanh', sn=sn)
    
    def forward(self, x):
        x = self.deconv1(x)  # (?, 128, 8, 8)
        x = self.deconv2(x)  # (?, 256, 16, 16)
        x = self.deconv3(x)  # (?, 3, 32, 32
        x = self.deconv4(x)
        return x

class FullyInvConvNet5_4(nn.Module):
    """Decoder"""
    def __init__(self, conv_dim=64, i_dim=1024, o_dim=1, size=32, bn=True, mid_ks=1, act=0, sn=False, **kwargs):
        super(FullyInvConvNet5_4, self).__init__()
        self.deconv1 = deconv(i_dim, 8*conv_dim, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=bn, act=act, sn=sn)
        self.deconv2 = deconv(8*conv_dim, 4*conv_dim, 4, bn=bn, act=act, sn=sn)
        self.deconv3 = deconv(4*conv_dim, 2*conv_dim, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=bn, act=act, sn=sn)
        self.deconv4 = deconv(2*conv_dim, conv_dim, 4, bn=bn, act=act, sn=sn)
        self.deconv5 = deconv(conv_dim, o_dim, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=False, act='tanh', sn=sn)
    
    def forward(self, x):
        x = self.deconv1(x)  # (?, 1024, 8, 8)
        x = self.deconv2(x)  # (?, 512, 16, 16)
        x = self.deconv3(x)  # (?, 256, 16, 16)
        x = self.deconv4(x)  # (?, 128, 32, 32)
        x = self.deconv5(x)  # (?, 3, 32, 32)
        return x


class FullyInvConvNet5(nn.Module):
    """Decoder"""
    def __init__(self, conv_dim=64, i_dim=1024, o_dim=1, size=32, bn=True, mid_ks=1, sn=False, **kwargs):
        super(FullyInvConvNet5, self).__init__()
        self.deconv1 = deconv(i_dim, conv_dim*2, 4, bn=bn, act=act, sn=sn)
        self.deconv2 = deconv(conv_dim*2, conv_dim*2, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=bn, act=act, sn=sn)
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4, bn=bn, act=act, sn=sn)
        self.deconv4 = deconv(conv_dim, conv_dim, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=bn, act=act, sn=sn)
        self.deconv5 = deconv(conv_dim, o_dim, 4, bn=False, act='tanh', sn=sn)
    
    def forward(self, x):
        x = self.deconv1(x)  # (?, 128, 8, 8)
        x = self.deconv2(x)  # (?, 128, 8, 8)
        x = self.deconv3(x)  # (?, 256, 16, 16)
        x = self.deconv4(x)  # (?, 256, 16, 16)
        x = self.deconv5(x)  # (?, 3, 32, 32)
        return x


class InvConvNet(nn.Module):
    def __init__(self, conv_dim=64, size=32, i_dim=1024, channels=1,
                 fullyconv=False, depth=4, strides=None):
        super(InvConvNet, self).__init__()
        if strides is None:
            strides = [2] * (depth + fullyconv - 1)
        else:
            depth = len(strides) + 1 - fullyconv
        self.conv_dim = conv_dim
        resize = int(np.prod(strides)) # exp(# conv layers)
        filters = conv_dim * resize//strides[-1]

        if fullyconv:
            self.add_module('deconv1', deconv(i_dim, filters, 4, stride=strides[0]))
        else:
            self._size = size // resize
            self.fc = linear(i_dim, filters * self._size**2)
            self.filters = filters
        for i, stride in enumerate(strides[fullyconv: -1]):
            name = 'deconv%d' % (i + 1 + fullyconv)
            self.add_module(name, deconv(filters, filters//stride, 4, stride=stride))
            filters = filters//stride
        self.add_module('deconv%d' % (depth + fullyconv - 1),
                        deconv(filters, channels, 4, stride=strides[-1], bn=False))

    def forward(self, x):
        if hasattr(self, 'fc'):
            x = self.fc(x)
            x = x.view(x.size(0), self.filters, self._size, self._size)
        i = 1
        while ('deconv%d' % i) in self._modules:
            layer = self._modules['deconv%d' % i]
            x = layer(x)
            i += 1
        return x

class MLP(nn.Module):
    """Discriminator"""
    def __init__(self, dim=64, i_dim=None, o_dim=1, bias=True, bn=False, layers=4, dropout=0,
                 act=0, sn=False, top_act=None, activation_penalty=False, **kwargs):
        self.activation_penalty = activation_penalty
        if i_dim is None:
            i_dim = dim
        assert dim//16 >= 1, 'dim = %d' % dim
        super(MLP, self).__init__()
        self.model =[]
        for _ in range(layers-1):
            self.model += [linear(i_dim, dim, bias=bias, bn=bn, act=act, sn=sn, dropout=dropout)]
            i_dim = dim
        self.model.append(linear(i_dim, o_dim, bias=bias, bn=False, act=top_act, sn=sn))
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        if self.activation_penalty:
            acts = []
            for layer in self._modules.values():
                x = layer(x)
                acts.append(x)
            return x, utils.activation_penalty(acts)
        return self.model(x)


class ConvNet2(nn.Module):
    def __init__(self, conv_dim=64, i_dim=3, o_dim=1, size=8, stride=2, activation_penalty=False,
                 bn=True, sn=False, act=0, top_activation=False, **kwargs):
        super(ConvNet2, self).__init__()
        self.activation_penalty = activation_penalty
        self.conv1 = conv(i_dim, conv_dim, 2+stride, stride=stride, bn=True, pad=stride-1, act=act, sn=sn)
        size_  = size - 2 if stride == 1 else size//stride
        self.fc = linear(conv_dim*size_**2, o_dim, bn=False, act=act if top_activation else None, sn=sn)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.fc(x1)
        if self.activation_penalty:
            return x2, utils.activation_penalty([x1, x2])
        return x2


class ConvNet2_1(nn.Module):
    def __init__(self, conv_dim=64, i_dim=3, o_dim=1, size=8, activation_penalty=False,
                 bn=True, sn=False, top_activation=False, act=0, **kwargs):
        super(ConvNet2_1, self).__init__()
        self.activation_penalty = activation_penalty
        self.bn0 = nn.BatchNorm2d(i_dim)
        self.act0 = nn.LeakyReLU(act) if (act > 0) else nn.ReLU()
        self.conv1 = conv(i_dim, conv_dim, 4, stride=2, bn=True, pad=1, act=act, sn=sn)
        self.fc1 = linear(conv_dim*(size//2)**2, 1024, bn=True, act=act, sn=sn)
        self.fc2 = linear(1024, o_dim, bn=False, act=act if top_activation else None, sn=sn)

    def forward(self, x):
        x1 = self.conv1(self.act0(self.bn0(x)))
        x1 = x1.view(x1.size(0), -1)
        x2 = self.fc1(x1)
        x3 = self.fc2(x2)
        if self.activation_penalty:
            return x3, utils.activation_penalty([x1, x2, x3])
        return x3


class ConvNet3b(nn.Module):
    def __init__(self, conv_dim=64, i_dim=3, o_dim=1, size=8, stride=1, activation_penalty=False,
                 act=0, sn=False, bn=True, **kwargs):
        super(ConvNet3, self).__init__()
        self.activation_penalty = activation_penalty
        self.conv1 = conv(i_dim, conv_dim, 2+stride, stride=stride, bn=False, pad=1, act=act, sn=sn)
        self.conv2 = conv(conv_dim, conv_dim*stride, 2+stride, stride=stride, bn=bn, pad=1, act=act, sn=sn)
        size_  = size//(stride**2)
        self.fc_dim = conv_dim*stride*size_**2
        self.fc = linear(self.fc_dim, o_dim, bn=False, act=None, sn=sn)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = x2.view(x2.size(0), self.fc_dim)
        x3 = self.fc(x2)
        if self.activation_penalty:
            return x3, utils.activation_penalty([x1, x2, x3])
        return x3


class ConvNet3(nn.Module):
    def __init__(self, conv_dim=64, i_dim=3, o_dim=1, size=8, stride=1, activation_penalty=False,
                 act=0, sn=False, bn=True, **kwargs):
        super(ConvNet3, self).__init__()
        self.activation_penalty = activation_penalty
        if size <= 4:
            size_ = size
            pad = 1
        else:
            size_  = size - 4 if stride == 1 else size//(stride**2)
            pad = stride - 1

        self.conv1 = conv(i_dim, conv_dim, 2+stride, stride=stride, bn=False, pad=pad, act=act, sn=sn)
        self.conv2 = conv(conv_dim, conv_dim*stride, 2+stride, stride=stride, bn=bn, pad=pad, act=act, sn=sn)
        self.fc_dim = conv_dim*stride*size_**2
        self.fc = linear(self.fc_dim, o_dim, bn=False, act=None, sn=sn)
        print('ConvNet3', conv_dim, stride, size, size_, self.fc_dim)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = x2.view(x2.size(0), self.fc_dim)
        x3 = self.fc(x2)
        if self.activation_penalty:
            return x3, utils.activation_penalty([x1, x2, x3])
        return x3


class ConvNet3fc2(nn.Module):
    def __init__(self, conv_dim=64, i_dim=3, o_dim=1, size=8, stride=1, activation_penalty=False,
                 act=0, sn=False, bn=True, **kwargs):
        super(ConvNet3fc2, self).__init__()
        self.activation_penalty = activation_penalty
        self.conv1 = conv(i_dim, conv_dim, 2+stride, stride=stride, bn=False, pad=stride-1, act=act, sn=sn)
        self.conv2 = conv(conv_dim, conv_dim*stride, 2+stride, stride=stride, bn=bn, pad=stride-1, act=act, sn=sn)
        size_  = size - 4 if stride == 1 else size//(stride**2)
        self.fc_dim = conv_dim*stride*size_**2
        self.fc = linear(self.fc_dim, 256, bn=bn, act=act, sn=sn)
        self.fc2 = linear(256, o_dim, bn=False, act=None, sn=sn)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = x2.view(x2.size(0), self.fc_dim)
        x3 = self.fc(x2)
        x4 = self.fc2(x3)
        if self.activation_penalty:
            return x4, utils.activation_penalty([x1, x2, x3, x4])
        return x4


class ConvNet5(nn.Module):
    conv_multiplier = 1
    """Discriminator"""
    def __init__(self, conv_dim=64, i_dim=3, o_dim=1, size=32, fc_bias=True,
        activation_penalty=False, act=0, bn=False, sn=False, top_ln=False, **kwargs):
        super(ConvNet5, self).__init__()
        conv_dim *= self.conv_multiplier
        self.activation_penalty = activation_penalty
        self.conv1 = conv(i_dim, conv_dim, 4, bn=False, act=act, sn=sn)
        self.conv2 = conv(conv_dim, conv_dim*2, 4, bn=bn, act=act, sn=sn)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4, bn=bn, act=act, sn=sn)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4, bn=bn, act=act, sn=sn)
        self.fc = linear(conv_dim * 8 * (size//16)**2, o_dim, bias=fc_bias, bn=False, act=None, sn=sn)
        if top_ln:
            self.ln = nn.LayerNorm(o_dim)
        else:
            self.ln = lambda x: x

    def forward(self, x):
        x1 = self.conv1(x)   # (?, 64, 16, 16)
        x2 = self.conv2(x1)  # (?, 128, 8, 8)
        x3 = self.conv3(x2)  # (?, 256, 4, 4)
        x4 = self.conv4(x3)  # (?, 256, 4, 4)
        x4 = x4.view(x4.size(0), -1)
        x5 = self.ln(self.fc(x4))
        if self.activation_penalty:
            return x5, utils.activation_penalty([x1, x2, x3, x4, x5])
        return x5

class ConvNet5x2(ConvNet5):
    conv_multiplier = 2


class ConvNet5fc2(ConvNet5):
    """Discriminator"""
    def __init__(self, conv_dim=64, o_dim=1, size=32, fc_bias=True, act=0, bn=False, sn=False, **kwargs):
        super(ConvNet5fc2, self).__init__(conv_dim=conv_dim, act=act, bn=bn, sn=sn, **kwargs)
        self.fc = linear(conv_dim * self.conv_multiplier * 8 * (size//16)**2,
                         256, bias=fc_bias, bn=bn, act=act, sn=sn)
        self.fc2 = linear(256, o_dim, bias=fc_bias, bn=False, act=None, sn=sn)

    def forward(self, x):
        x = super(ConvNet5fc2, self).forward(x)
        if self.activation_penalty:
            x6, act5 = self.fc2(x[0]), x[1]
            return x6, act5 + utils.activation_penalty([x6])
        return self.fc2(x)

class ConvNet5fc2x2(ConvNet5fc2):
    conv_multiplier = 2


class FullyConvNet1(nn.Module):
    def __init__(self, conv_dim=64, i_dim=3, o_dim=1, activation_penalty=False, act=0, 
                 sn=False, ks=4, **kwargs):
        super(FullyConvNet1, self).__init__()
        self.activation_penalty = activation_penalty
        self.conv1 = conv(i_dim, o_dim, ks, stride=2, bn=False, pad=(ks-1)//2, act=act, sn=sn)

    def forward(self, x):
        x = self.conv1(x)
        if self.activation_penalty:
            return x, utils.activation_penalty([x])
        return x


class FullyConvNet2(nn.Module):
    def __init__(self, conv_dim=64, i_dim=3, o_dim=1, activation_penalty=False, bn=True,
                 ks=4, top_activation=True, act=0, sn=False, **kwargs):
        super(FullyConvNet2, self).__init__()
        self.activation_penalty = activation_penalty
        self.conv1 = conv(i_dim, conv_dim, ks, stride=2, bn=bn, pad=(ks-1)//2, act=act, sn=sn)
        self.conv2 = conv(conv_dim, o_dim, ks, stride=2, bn=False, pad=(ks-1)//2, act=act if top_activation else None, sn=sn)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if self.activation_penalty:
            return x2, utils.activation_penalty([x1, x2])
        return x2
        
class FullyConvNet2_2(FullyConvNet2):
    def __init__(self, conv_dim=64, i_dim=3, o_dim=1, activation_penalty=False, bn=True,
                 ks=4, top_activation=True, act=0, sn=False, **kwargs):
        super(FullyConvNet2, self).__init__()
        self.activation_penalty = activation_penalty
        self.conv1 = conv(i_dim, conv_dim, 4, stride=2, bn=bn, pad=1, act=act, sn=sn)
        self.conv2 = conv(conv_dim, o_dim, ks, stride=1, bn=False, pad=(ks-1)//2, act=act if top_activation else None, sn=sn)


class FullyConvNet3_2(nn.Module):
    def __init__(self, conv_dim=64, i_dim=3, o_dim=1, activation_penalty=False, bn=True,
                 ks=4, top_activation=True, act=0, sn=False, top_sn=False, res_blocks=0, **kwargs):
        super(FullyConvNet3_2, self).__init__()
        self.activation_penalty = activation_penalty
        self.conv1 = conv(i_dim, conv_dim, 4, stride=2, bn=bn, pad=1, act=act, sn=False)

        if res_blocks > 0:
            self.mid = ResBlocks(n_blocks=res_blocks, dim=conv_dim, bn=bn, act=act)
        else:
            self.mid = conv(conv_dim, conv_dim, 1, stride=1, bn=bn, pad=0, act=act, sn=sn)

        self.conv3 = conv(conv_dim, o_dim, 3, stride=1, bn=False, pad=1, act=act if top_activation else None, sn=top_sn)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.mid(x1)
        x3 = self.conv3(x2)
        if self.activation_penalty:
            return x3, utils.activation_penalty([x1, x2, x3])
        return x3


class FullyConvNet3_4(FullyConvNet3_2):
    def __init__(self, conv_dim=64, i_dim=3, o_dim=1, activation_penalty=False, bn=True,
                 ks=4, top_activation=True, act=0, sn=False, top_sn=False, res_blocks=0, **kwargs):
        super(FullyConvNet3_4, self).__init__()
        self.activation_penalty = activation_penalty
        self.conv1 = conv(i_dim, conv_dim, ks, stride=2, bn=bn, pad=(ks-1)//2, act=act, sn=sn)

        if res_blocks > 0:
            self.mid = ResBlocks(n_blocks=res_blocks, dim=conv_dim, bn=bn, act=act)
        else:
            self.mid = conv(conv_dim, conv_dim, 1, stride=1, bn=bn, pad=0, act=act, sn=sn)

        self.conv3 = conv(conv_dim, o_dim, ks, stride=2, bn=False, pad=(ks-1)//2, act=act if top_activation else None, sn=top_sn)


class FullyConvNet3_8(FullyConvNet3_2):
    def __init__(self, conv_dim=64, i_dim=3, o_dim=1, activation_penalty=False, bn=True,
                 ks=4, top_activation=True, act=0, sn=False, top_sn=False, res_blocks=0, **kwargs):
        super(FullyConvNet3_8, self).__init__()
        self.activation_penalty = activation_penalty

        if res_blocks > 0:
            self.conv1 = conv(i_dim, 2*conv_dim, ks, stride=2, bn=bn, pad=(ks-1)//2, act=act, sn=sn)
            self.mid = ResBlocks(n_blocks=res_blocks - 1, down_blocks=1, dim=2*conv_dim, bn=bn, act=act)
        else:
            self.conv1 = conv(i_dim, conv_dim, ks, stride=2, bn=bn, pad=(ks-1)//2, act=act, sn=sn)
            self.mid = conv(conv_dim, 2*conv_dim, ks, stride=2, bn=bn, pad=(ks-1)//2, act=act, sn=sn)

        self.conv3 = conv(2*conv_dim, o_dim, ks, stride=2, bn=False, pad=(ks-1)//2, act=act if top_activation else None, sn=top_sn)


class OffsetConvNet3_2(FullyConvNet3_2):
    def __init__(self, conv_dim=64, i_dim=3, o_dim=1, offset_ks=4, **kwargs):
        super(OffsetConvNet3_2, self).__init__(conv_dim=conv_dim, i_dim=i_dim, 
                                               o_dim=o_dim, **kwargs)
        assert offset_ks % 2 == 1, 'Only odd offset_ks supported'
        kernel_dim = offset_ks**2
        self.kernel_conv1 = conv(o_dim, kernel_dim, 4, stride=4, pad=0, **kwargs)
        self.kernel_conv2 = conv(kernel_dim, kernel_dim, 4, strdie=4, pad=0, **kwargs)
        self.offset = OffsetConv2d(offset_ks, padding=(offset_ks-1)//2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        k1 = self.kernel_conv1(x3)
        k2 = self.kernel_conv2(k1).view(x.size(0), -1)
        x4 = self.offset(x3, k2)
        if self.activation_penalty:
            return x4, utils.activation_penalty([x1, x2, x3, x4])
        return x4


class FullyConvNet3(nn.Module):
    def __init__(self, conv_dim=64, size=32, i_dim=3, o_dim=1, mid_ks=3, bn=True, act=0, sn=False, **kwargs):
        super(FullyConvNet3, self).__init__()
        self.conv1 = conv(i_dim, conv_dim, 4, bn=False, act=act, sn=sn)
        self.conv2 = conv(conv_dim, conv_dim, mid_ks, stride=1, pad=(mid_ks - 1)//2, bn=bn, act=act, sn=sn)
        self.conv3 = conv(conv_dim, o_dim, 4, bn=bn, act=act, sn=sn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class FullyConvNet4(nn.Module): # overall repective field at output = 16x16
    def __init__(self, conv_dim=64, size=32, i_dim=1, o_dim=1, mid_ks=3, bn=True, act=0, sn=False, **kwargs):
        super(FullyConvNet4, self).__init__()
        self.conv1 = conv(i_dim, conv_dim, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=False, act=act, sn=sn)
        self.conv2 = conv(conv_dim, conv_dim*2, 4, bn=bn, act=act, sn=sn)
        self.conv3 = conv(conv_dim*2, conv_dim*4, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=bn, act=act, sn=sn)
        self.conv4 = conv(conv_dim*4, o_dim, 4, bn=bn, act=act, sn=sn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class FullyConvNet5_4(nn.Module): # overall repective field at output = 16x16
    def __init__(self, conv_dim=64, size=32, i_dim=1, o_dim=1, mid_ks=3, bn=True, act=0, sn=False, **kwargs):
        super(FullyConvNet5_4, self).__init__()
        self.conv1 = conv(i_dim, conv_dim, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=False, act=act, sn=sn)
        self.conv2 = conv(conv_dim, conv_dim*2, 4, bn=bn, act=act, sn=sn)
        self.conv3 = conv(conv_dim*2, conv_dim*2, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=bn, act=act, sn=sn)
        self.conv4 = conv(conv_dim*2, conv_dim*4, 4, bn=bn, act=act, sn=sn)
        self.conv5 = conv(conv_dim*4, o_dim, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=bn, sn=sn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class FullyConvNet5(nn.Module):
    def __init__(self, conv_dim=64, size=32, i_dim=1, o_dim=1, mid_ks=3, bn=True, act=0, sn=False, **kwargs):
        super(FullyConvNet5, self).__init__()
        self.conv1 = conv(i_dim, conv_dim, 4, bn=False, act=act, sn=sn)
        self.conv2 = conv(conv_dim, conv_dim, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=bn, act=act, sn=sn)
        self.conv3 = conv(conv_dim, conv_dim*2, 4, bn=bn, act=act, sn=sn)
        self.conv4 = conv(conv_dim*2, conv_dim*2, mid_ks, stride=1, pad=(mid_ks-1)//2, bn=bn, act=act, sn=sn)
        self.conv5 = conv(conv_dim*2, o_dim, 4, bn=bn, act=act, sn=sn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class NarrowConvNet4(nn.Module):
    """Discriminator"""
    def __init__(self, conv_dim=64, i_dim=1, o_dim=1, size=4, act=0, sn=False, **kwargs):
        super(NarrowConvNet4, self).__init__()
        self.conv1 = conv(i_dim, conv_dim, 3, stride=1, bn=False, act=act, sn=sn)
        self.conv2 = conv(conv_dim, conv_dim, 3, stride=1, act=act, sn=sn)
        self.conv3 = conv(conv_dim, conv_dim, 3, stride=1, act=act, sn=sn)
        self.fc = linear(conv_dim*size**2, o_dim, bias=True, bn=False, act=None, sn=sn)

    def forward(self, x):
        x = self.conv1(x)  # (?, 64, 4, 4)
        x = self.conv2(x)  # (?, 64, 4, 4)
        x = self.conv3(x)  # (?, 64, 4, 4)
        x = x.view(x.size(0), -1)             # (?, 1024)
        x = self.fc(x)
        return x 


class ConvNet(nn.Module):
    """Encoder"""
    def __init__(self, conv_dim=64, size=32, o_dim=1, channels=1, 
                 fullyconv=False, depth=4, strides=None, kernel_size=4):
        super(ConvNet, self).__init__()
        if strides is None:
            strides = [2] * (depth -1 + fullyconv)
        else:
            depth = len(strides) + 1 - fullyconv
        
        self.convs = []
        _dim, dim_ = channels, conv_dim
        for i, stride in enumerate(strides[:depth - 1]):
            name = 'conv' + str(i+1)
            self.add_module(name, conv(_dim, dim_, kernel_size, bn=(i!=0), stride=stride))
            self.convs.append(name)
            _dim, dim_ = dim_, dim_ * stride
            size = size // stride
        if fullyconv:
            self.add_module('conv%d' % depth, conv(_dim, o_dim, kernel_size, stride=stride))
            self.convs.append('conv%d' % depth)
        else:
            self.fc = nn.Linear(_dim * size * size, o_dim)

    
    def forward(self, x):
        for name in self.convs:
            layer = self._modules[name]
            x = F.leaky_relu(layer(x), 0.2)
        if hasattr(self, 'fc'):
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class MINE1x1Conv(nn.Module):
    def __init__(self, conv_dim=64, i_dim=1, o_dim=1, layers=3, initial_activation=False, act=0, sn=False, **kwargs):
        super(MINE1x1Conv, self).__init__()
        model = get_act(act) if initial_activation else []
        for _ in range(layers-1):
            model += [conv(i_dim, conv_dim, 1, stride=1, pad=0, bn=False, bias=True, act=act, sn=sn)]
            i_dim = conv_dim
        model += [conv(i_dim, o_dim, 1, stride=1, pad=0, bn=False, bias=True, act=None, sn=sn)]
#        model += [nn.ReLU()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class MINEImageVector(nn.Module):
    def __init__(self, z_dim, x_dim, x_size, conv_dim=64, bn=False, act=0, **kwargs):
        super(MINEImageVector, self).__init__()
        #print('MINE I V ', z_dim, x_dim, x_size, (x_size//2)**2 * x_dim * 2)
        #size = (z0.size(0), 2*x_dim, x_size, x_size)
        self.z_fc = linear(z_dim, (x_size//2)**2 * x_dim*2, bn=False, act=act)
        self.x_conv = conv(x_dim, 2*x_dim, 4, stride=2, pad=1, bn=False, bias=True, act=act)
        self.conv1 = conv(4*x_dim, conv_dim, 4, stride=2, pad=1, bn=bn, bias=True, act=act)
        self.conv2 = conv(conv_dim, 2*conv_dim, 4, stride=2, pad=1, bn=bn, bias=True, act=act)
        self.conv3 = conv(2*conv_dim, 4*conv_dim, 4, stride=2, pad=1, bn=bn, bias=True, act=act)
        self.top_fc = nn.Linear(4*conv_dim*(x_size//16)**2, 1, bias=True)

    def forward(self, z, x):
        #print('MINE I V', x.size(), z.size())
        x0 = self.x_conv(x)
        z0 = self.z_fc(z)
        z0 = z0.view(x0.size())
        out = self.conv1(torch.cat((x0, z0), dim=1))
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view((out.size(0), -1))
        out = self.top_fc(out)
        return out

class MINEImageVector2(nn.Module):
    def __init__(self, z_dim, x_dim, x_size, conv_dim=64, bn=False, act=0, sn=False, **kwargs):
        super(MINEImageVector2, self).__init__()
        x_o_dim = conv_dim * 4 * (x_size//8)**2
        self.conv_net = FullyConvNet5(i_dim=x_dim, conv_dim=conv_dim, o_dim=conv_dim*4, bn=bn, act=act, sn=sn)
        self.mlp = MLP(i_dim=x_o_dim + z_dim, dim=conv_dim*16, o_dim=1, bn=bn, layers=3, act=act, sn=sn)

    def forward(self, z, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        out = self.mlp(torch.cat((x, z), dim=1))
        return out


class ConstSizeResNet4(nn.Module):
    def __init__(self, conv_dim=64, size=8, i_dim=1, o_dim=1, kernel_size=3):
        super(ConstSizeResNet4, self).__init__()
        self.res1 = ResidualBlock(i_dim, conv_dim, kernel_size, hw=size)
        self.res2 = ResidualBlock(conv_dim, conv_dim, kernel_size, hw=size)
        self.res3 = ResidualBlock(conv_dim, conv_dim, kernel_size, hw=size)
        self.res4 = ResidualBlock(conv_dim, o_dim, kernel_size, hw=size)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return x


class SamEncoder(nn.Module):
    def __init__(self, i_dim=1, conv_dim=64, o_dim=128, act=0, bn=True, sn=False, **kwargs):
        super(SamEncoder, self).__init__()
        self.conv1 = conv(i_dim, conv_dim, 7, stride=1, pad=3, bias=False, act=act, bn=bn, sn=sn)
        self.conv2 = conv(conv_dim, 2*conv_dim, 4, stride=2, pad=1, bias=False, act=act, bn=bn, sn=sn)
        self.conv3 = conv(2*conv_dim, o_dim, 3, stride=1, pad=1, bias=False, act=None, bn=bn, sn=sn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SamGenerator(nn.Module):
    def __init__(self, i_dim=64, conv_dim=64, o_dim=1, act=0, top_act='tanh', bn=True, other_return=None, sn=False, **kwargs):
        super(SamGenerator, self).__init__()
        self.other_return = other_return
        self.deconv = deconv(i_dim, conv_dim, 4, stride=2, pad=1, bias=False, bn=bn, act=act, sn=sn)
        self.conv1 = conv(conv_dim, conv_dim, 3, stride=1, pad=1, bias=False, bn=bn, act=act, sn=sn)
        self.pad = nn.ReflectionPad2d((3, 3, 3, 3))
        self.conv2 = conv(conv_dim, o_dim, 7, stride=1, pad=0, bias=False, act=top_act, bn=False, sn=sn)

    def forward(self, x):
        x1 = self.deconv(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(self.pad(x2))
        if self.other_return == 1:
            return x3, x1
        if self.other_return == 2:
            return x3, x2
        if self.other_return == 3:
            return x3, x3
        return x3


class SamDiscriminator(nn.Module):
    def __init__(self, i_dim=1, conv_dim=64, o_dim=1, act=0, size=32, bn=True,
                 activation_penalty=False, additional_dim=0, sn=False, **kwargs):
        super(SamDiscriminator, self).__init__()
        self.activation_penalty = activation_penalty
        self.conv1 = conv(i_dim, conv_dim, 7, stride=1, pad=3, bias=False, bn=bn, act=act, sn=sn)
        self.conv2 = conv(conv_dim, 2*conv_dim, 4, stride=2, pad=1, bias=False, bn=bn, act=act, sn=sn)
        self.conv3 = conv(2*conv_dim, 4*conv_dim, 4, stride=2, pad=1, bias=False, bn=bn, act=act, sn=sn)
        i_dim_mlp = (size//4)**2 * 4*conv_dim + additional_dim
        self.mlp = MLP(i_dim=i_dim_mlp, dim=8*conv_dim, o_dim=1, layers=3, act=act,
                       bn=False, activation_penalty=activation_penalty, sn=sn)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = x3.view(x3.size(0), -1)
        if self.activation_penalty:
            x4, act4 = self.mlp(x3)
            act = utils.activation_penalty([x1, x2, x3]) + act4
            return x4, act
        return self.mlp(x3)

class SamMINE(SamDiscriminator):
    def __init__(self, z_dim=64, x_dim=64, x_size=32, **kwargs):
        super(SamMINE, self).__init__(i_dim=x_dim, o_dim=1,  size=x_size, additional_dim=z_dim, **kwargs)

    def forward(self, z, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat((x3.view(x3.size(0), -1), z), dim=1)
        if self.activation_penalty:
            out, act_out = self.mlp(out)
            act_out += utils.activation_penalty([x1, x2, x3])
            return out, act_out
        return self.mlp(out)


class ConvNet5MINE(ConvNet5):
    def __init__(self, z_dim=64, x_dim=64, x_size=32, **kwargs):
        self.size = x_size
        super(ConvNet5MINE, self).__init__(i_dim=x_dim + z_dim, size=x_size, **kwargs)

    def forward(self, z, x):
        z = z.repeat(self.size, self.size, 1, 1).permute(2, 3, 0, 1)
        z = z.view(z.size(0), z.size(1), self.size, self.size)
        return super(ConvNet5MINE, self).forward(torch.cat((x, z), dim=1))


class SamXY(SamDiscriminator):
    def __init__(self, **kwargs):
        super(SamXY, self).__init__(o_dim=1, **kwargs)

    def forward(self, x, y):
        return super(SamXY, self).forward(torch.cat((x, y), 1))


class CPCEncoder(nn.Module):
    def __init__(self, i_dim=1, conv_dim=64, o_dim=None, bn=True, sn=False, act=0, dropout=.1, **kwargs):
        super(CPCEncoder, self).__init__()
        self.conv1 = conv(i_dim, conv_dim, 4, stride=2, pad=0, bn=bn, sn=sn, act=act)
        self.conv2 = conv(conv_dim, conv_dim, 1, stride=1, pad=0, bn=bn, sn=sn, act=act)
        if o_dim is None:
            o_dim = conv_dim
        self.conv3 = conv(conv_dim, o_dim, 3, stride=2, pad=0, bn=False, sn=sn, act=None)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x3


class CPCRes3Encoder(nn.Module):
    def __init__(self, i_dim=1, conv_dim=64, o_dim=None, act=0., **kwargs):
        super(CPCRes3Encoder, self).__init__()
        self.f = Unfold(8, stride=4)
        self.res1 = ResidualBlock(i_dim, conv_dim, None, size=8, act=act)
        self.res2 = ResidualBlock(conv_dim, conv_dim, None, size=8, act=act)
        self.res3 = ResidualBlock(conv_dim, 2*conv_dim, 'down', size=8, act=act)
        self.conv = conv(2*conv_dim, o_dim, 4, pad=0, act=None)

    def forward(self, x):
        bs = x.size(0)
        x0 = self.f(x)
        s = x0.size(2)
        x1 = self.res1(x0)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.conv(x3)
        x4 = x4.view(bs, 7, 7, -1).permute(0, 3, 1, 2).contiguous()
        return x4



class CPCNoise(nn.Module):
    def __init__(self, o_dim=64, noise_dim=64, **kwargs):
        super(CPCNoise, self).__init__()
        self.linear = linear(noise_dim, o_dim*5*5, **kwargs)
        self.deconv = deconv(o_dim, o_dim, 3, stride=1, pad=0, **kwargs)
        self.o_dim = o_dim

    def forward(self, x, z):
        z = self.linear(z)
        z = z.view(-1, self.o_dim, 5, 5)
        z = self.deconv(z)
        return torch.cat([x, z], 1)


class CPCGenerator(nn.Module):
    def __init__(self, i_dim, conv_dim, o_dim, sn=True, top_act='tanh', **kwargs):
        super(CPCGenerator, self).__init__()
        self.deconv1 = deconv(i_dim, conv_dim, 3, stride=2, pad=0, sn=sn, **kwargs)
        self.deconv2 = deconv(conv_dim, conv_dim, 1, stride=1, pad=0, sn=sn, **kwargs)
        self.deconv3 = deconv(conv_dim, o_dim, 4, stride=2, pad=0, sn=sn, bn=False, act=top_act)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class CPCGenerator5(nn.Module):
    def __init__(self, i_dim, conv_dim, o_dim, sn=True, top_act='tanh', **kwargs):
        super(CPCGenerator5, self).__init__()
        self.conv1 = conv(i_dim, 2*conv_dim, 3, stride=1, pad=1, sn=sn, **kwargs)
        self.deconv2 = deconv(2*conv_dim, conv_dim, 3, stride=2, pad=0, sn=sn, **kwargs)
        self.conv3 = conv(conv_dim, conv_dim, 1, stride=1, pad=0, sn=sn, **kwargs)
        self.conv4 = conv(conv_dim, conv_dim, 3, stride=1, pad=1, sn=sn, **kwargs)
        self.deconv5 = deconv(conv_dim, o_dim, 4, stride=2, pad=0, sn=sn, bn=False, act=top_act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.deconv5(x)
        return x



class MEncoder(nn.Module):
    def __init__(self, i_dim=1, conv_dim=64, o_dim=512, bn=False, downsample=3, res_blocks=2, **kwargs):
        super(MEncoder, self).__init__()
        print('Initializing MEncoder...')
        layers = [conv(i_dim, conv_dim, 3, stride=1, pad=1, bn=False, **kwargs)]
        dim = conv_dim
        print('Initializing MEncoder 1 ...')
        for _ in range(downsample):
            print(_)
            layers += [conv(dim, dim*2, 4, bn=bn, **kwargs)]
            dim *= 2
        print('Initializing MEncoder 2 ...')
        layers += ResBlocks(n_blocks=res_blocks, return_list=True, dim=dim, bn=bn, **kwargs)
        print('Initializing MEncoder 3...')
        if dim != o_dim:
            layers += [conv(dim, o_dim, 1, stride=1, pad=0, bn=bn, **kwargs)]
        print('Initializing MEncoder 4 ...')
        self.model = nn.Sequential(*layers)
        print('Initializing MEncoder...DOne')

    def forward(self, x):
        return self.model(x)


class MGenerator(nn.Module):
    def __init__(self, i_dim=512, conv_dim=64, o_dim=1, bn=False, upsample=3, res_blocks=2, top_act='tanh', **kwargs):
        super(MGenerator, self).__init__()
        layers = []
        dim = conv_dim * (2 ** upsample)
        if i_dim != dim:
            layers += [conv(i_dim, dim, 1, stride=1, pad=0, bn=bn, **kwargs)]
        layers += ResBlocks(n_blocks=res_blocks, return_list=True, dim=dim, bn=bn, **kwargs)
        for _ in range(upsample):
            layers += [deconv(dim, dim//2, 4, bn=bn, **kwargs)]
            dim = dim//2
        layers += [conv(dim, o_dim, 3, stride=1, pad=1, bn=False, act=top_act)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CPCRNN(nn.Module):
    def __init__(self, dim, dropout=.1, **kwargs):
        super(CPCRNN, self).__init__()
        self.gru = torch.nn.GRU(input_size=dim, hidden_size=dim, dropout=dropout)

    def forward(self, x):
        s = x.size()
        x = x.view(s[0], s[1], -1) # 7x7 grid flatten
#        print('RRN', x.size())
        x = x.permute(2, 0, 1) # seq dim to 2nd dimention, features 3rd
#        print('RRN', x.size())
        x1, h_n = self.gru(x)
#        print('RRN', x1.size(), h_n.size())
        x1 = x1.permute(1, 2, 0)
#        print('RRN', x1.size(), h_n.size())
        x1, h_n = x1.view(s), h_n.view(s[0], s[1])
#        print('RRN', x1.size(), h_n.size())
        return x1, h_n


class CPCBilinear(nn.Module):
    def __init__(self, dim, k):
        super(CPCBilinear, self).__init__()
        self.dim = dim
        for j in range(1, 1+k):
            self.add_module('W' + str(j), nn.Bilinear(dim, dim, 1, bias=False))

    def forward(self, c, z, k):
        W = getattr(self, 'W' + str(k))
#        print('BILINEAR', c.size(), z.size(), self.dim, k)
        s = c.size()
        c = c.permute(0, 2, 3, 1).contiguous().view(-1, s[1])
        z = z.permute(0, 2, 3, 1).contiguous().view(-1, s[1])
#        print('BILINEAR', c.size(), z.size(), self.dim)
        x = W(c, z)
#        print('BILINEAR', x.size())
        x = x.view(s[0], 1, s[2], s[3])
#        print('BILINEAR', x.size())
        return x


