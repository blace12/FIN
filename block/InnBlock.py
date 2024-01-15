import torch
import torch.nn as nn
from block.DenseNet import ResidualDenseBlock_out as DB
from block.Encoder import Encoder
from block.Decoder import Decoder


class Noise_INN_block(nn.Module):
    def __init__(self, clamp=2.0):
        super().__init__()

        self.clamp = clamp
        self.r = DB(input=3, output=9)
        self.y = DB(input=3, output=9)
        self.f = DB(input=9, output=3)


    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        x1, x2 = x[0], x[1]

        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2

            s1, t1 = self.r(y1), self.y(y1)

            y2 = torch.exp(s1) * x2 + t1

            out = [y1, y2]

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / torch.exp(s1)

            t2 = self.f(y2)
            y1 = x1 - t2

            out = [y1, y2]
        return out

class INN_block(nn.Module):
    def __init__(self, clamp=2.0):
        super().__init__()

        self.clamp = clamp
        self.r = Decoder()
        self.y = Decoder()
        self.f = Encoder()
        
    
    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))


    def forward(self, x, rev=False):
        # x[0]为原始图像，x[1]为水印信息, x[2]为上一层的水印信息
        x1, x2 = x[0], x[1]
        if len(x)==3:
            x3 = x[2]
        else:
            # x3 = torch.zeros(x1.shape).cuda()
            x3 = torch.zeros(x1.shape)

        if not rev:
            # x2(batch_size*64)经过f函数编码后变成(batch_size*1*128*128)可以直接叠加在原始图像上
            t2 = self.f(x2)
            y1 = x1 + t2 + x3

            s1, t1 = self.r(y1), self.y(y1)

            y2 = torch.exp(s1) * x2 + t1

            out = [y1, y2, t2]

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / torch.exp(s1)

            t2 = self.f(y2)
            y1 = (x1 - t2 - x3)

            out = [y1, y2, t2]
        return out