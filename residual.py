from torch import nn
from residual_block import UpSampleConv

def get_act(act):
    if act is not None:
        if act == 'tanh':
            return nn.Tanh()
        if act == 0:
            return nn.ReLU()
        if act > 0:
            return nn.LeakyReLU(act)
        raise Exception('wrong activation: ' + str(act))
    return None

class ResidualBlock(nn.Module):
    def __init__(self, i_dim, o_dim, resample=None, size=32, ks=3, act=0, top_act='same'):
#        print('RES', i_dim, o_dim, resample, size, ks, act, top_act)
        super(ResidualBlock, self).__init__()
#        print('RES', ks)
        if resample == 'down':
            self.norm1 = nn.LayerNorm([i_dim, size, size])
            self.norm2 = nn.LayerNorm([o_dim, size//2, size//2])
        elif (resample == 'up') or (resample is None):
            self.norm1 = nn.BatchNorm2d(o_dim if resample else i_dim)
            self.norm2 = nn.BatchNorm2d(o_dim)
        else:
            raise Exception('invalid resample value: ' + str(resample))
#        print('RES', ks)

        self.act = get_act(act)
        if top_act == 'same':
            self.top_act = self.act
        else:
            self.top_act = get_act(top_act)

        pad = (ks - 1) // 2

#        print('RES', ks)
        if resample == 'down':
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                nn.Conv2d(i_dim, o_dim, 1, 1, 0),
                nn.BatchNorm2d(o_dim)
            )
            self.conv1 = nn.Conv2d(i_dim, i_dim, ks, 1, pad, bias=False)
            self.conv2 = nn.Conv2d(i_dim, o_dim, ks, 2, pad, bias=True)
        elif resample == 'up':
            self.shortcut = nn.Sequential(
                UpSampleConv(i_dim, o_dim, 1),
                nn.BatchNorm2d(o_dim)
            )
            self.conv1 = UpSampleConv(i_dim, o_dim, ks)
            self.conv2 = nn.Conv2d(o_dim, o_dim, ks, 1, pad)
        elif resample is None:
            if i_dim != o_dim:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(i_dim, o_dim, 1, 1, 0),
                    nn.BatchNorm2d(o_dim)
                )
#            print('RES', ks)
            self.conv1 = nn.Conv2d(i_dim, i_dim, ks, 1, pad, bias=False)
            self.conv2 = nn.Conv2d(i_dim, o_dim, ks, 1, pad)
        else:
            raise Exception('invalid resample value: ' + str(resample))

    def forward(self, x):
        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.top_act:
            x = self.top_act(x + shortcut)
        return x

