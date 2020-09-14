# freda (todo) : 

import torch.nn as nn
import torch
import numpy as np 

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    return torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,\
    stride=stride, padding=padding, dilation=dilation, bias=True), nn.LeakyReLU(0.1))

def conv3d(in_channels, out_channels, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), dilation=1):
    return torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,\
    stride=stride, padding=padding, dilation=dilation, bias=True), nn.LeakyReLU(0.1))

def predict_flow(in_channels, out_channels=2):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

def predict_res(in_channels, out_channels=3):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


def warp(tensorInput, tensorFlow):
    torchHorizontal = torch.linspace(-1.0, 1.0, tensorInput.size(3)).view(1, 1, 1, tensorInput.size(3)).expand(tensorInput.size(0), 1, tensorInput.size(2), tensorInput.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, tensorInput.size(2)).view(1, 1, tensorInput.size(2), 1).expand(tensorInput.size(0), 1, tensorInput.size(2), tensorInput.size(3))
    tensorGrid = torch.cat([ torchHorizontal, torchVertical ], 1).cuda()
        # end
    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')


class DenseBlock(torch.nn.Module):
    def __init__(self, nif= 32, nof=32):
        super(DenseBlock, self).__init__()
        
        self.conv0 = conv(nif, nif)
        self.conv1 = conv(nif*2, nof)


    def forward(self,x):
        x0 = x
        x1 = self.conv0(x0)
        x2 = torch.cat((x1, x0), 1)
        out = self.conv1(x2)
        
        return out



class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):
    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


def init_deconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i,j,:,:] = torch.from_numpy(bilinear)


def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad
    return hook



'''
def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad
    return hook
import torch
from channelnorm_package.modules.channelnorm import ChannelNorm 
model = ChannelNorm().cuda()
grads = {}
a = 100*torch.autograd.Variable(torch.randn((1,3,5,5)).cuda(), requires_grad=True)
a.register_hook(save_grad(grads, 'a'))
b = model(a)
y = torch.mean(b)
y.backward()

'''

