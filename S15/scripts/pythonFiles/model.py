import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from IPython.core.debugger import set_trace

from torchvision.utils import make_grid
from torchsummary import summary
from PIL import Image
# New 
from torchvision import models
from zipfile import ZipFile
from pathlib import Path
import os ,sys, gc ,tarfile ,zipfile,pickle,pdb
from pprint import pprint

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
from zipfile import ZipFile


class Downsize (nn.Module):
    def __init__(self, inchannels, outchannels):
        super (Downsize,self).__init__()
        self.c1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding = 1, bias=False)
        self.conv11 = nn.Conv2d(outchannels, outchannels, kernel_size=1, stride=1, bias=False)
        self.bn1=nn.BatchNorm2d(outchannels)
        self.relu=nn.ReLU()

        self.c2= nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=2, padding = 1, bias=False)
        self.conv21 = nn.Conv2d(outchannels, outchannels, kernel_size=1, stride=1, bias=False)
        self.bn2=nn.BatchNorm2d(outchannels)

        self.dconv = nn.ConvTranspose2d(inchannels, outchannels,kernel_size=1, stride=2, bias=False)
    def forward(self,x):
        idchannel=self.dconv(x)
        x1=self.c1(x)
        x2=self.conv11(x1)
        x3=self.bn1(x2)
        x4=self.relu(x3)
        x5=self.c2(x4)
        x6=self.conv21(x5)
        x7=self.bn2(x6)
        x8=x7
        #x8=x7+idchannel
        x9=self.relu(x8)
        return x9


class Upsize(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(Upsize, self).__init__()
        self.c1=nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1,bias=False)
        self.conv11=nn.Conv2d(outchannels, outchannels, kernel_size=1, bias=False)
        self.bn1=nn.BatchNorm2d(outchannels)
        self.relu=nn.ReLU()
        self.c2=nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1,bias=False)
        self.conv21=nn.Conv2d(outchannels, outchannels, kernel_size=1, bias=False)
        self.bn2=nn.BatchNorm2d(outchannels)

        self.dconv=nn.ConvTranspose2d(inchannels, outchannels,kernel_size=3, padding=1, bias=False)

    def forward(self,x):
        idchannel=self.dconv(x)
        x1=self.c1(idchannel)
        x2=self.conv11(x1)
        x3=self.bn1(x2)
        x4=self.relu(x3)
        x5=self.c2(x4)
        x6=self.conv21(x5)
        x7=self.bn2(x6)
        x8=self.relu(x7)
        return x8
    
    
class DepthMask(nn.Module):
    def __init__(self):
        super(DepthMask, self).__init__()

        self.c1=nn.Sequential(
            nn.Conv2d(6,64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool=nn.MaxPool2d(3, stride=2, padding=1)

        self.fdm1=Downsize(64,128)
        self.fdm2=Downsize(128,256)
        self.fdm3=Downsize(256,512)
        self.fdm4=Upsize(512,256)
        self.fdm5=Upsize(256,128)
        self.fdm6=Upsize(128,64)
        self.fdm7=Upsize(64,64)
        self.fdm8=Upsize(64,32)
        self.lasconv=nn.Sequential(
            nn.Conv2d(32,1,kernel_size=1, bias=False)
        )
        self.fdm4v=Upsize(512,256)
        self.fdm5v=Upsize(256,128)
        self.fdm6v=Upsize(128,64)
        self.fdm7v=Upsize(64,64)
        self.fdm8v=Upsize(64,32)
        self.lastconvd=nn.Sequential(
            #nn.Conv2d(32,3,kernel_size=1, stride=1)
            nn.Conv2d(32,1,kernel_size=1, stride=1)
        )

    def forward (self, x):
   
        #z = torch.cat([sample['bg'], sample['bgfg']], dim=1)
        #z=torch.cat([x,y], dim=1)
        x0=self.c1(x)
        x1=self.maxpool(x0)
        x2=self.fdm1(x1)
        x3=self.fdm2(x2)
        x4=self.fdm3(x3)

        x4=nn.functional.interpolate(x4, scale_factor=4, mode='bilinear')
        x5=self.fdm4(x4)
        #print('x3:',x3.shape)
        #print('x4:',x4.shape)
        x5+=x3
        x5=nn.functional.interpolate(x5, scale_factor=7, mode='bilinear')
        x6=self.fdm5(x5)
        x2=nn.functional.interpolate(x2, scale_factor=2, mode='bilinear')
        #print('x6:',x6.shape)
        #print('x2:',x2.shape)
        
        x6+=x2
        x6=nn.functional.interpolate(x6, scale_factor=2, mode='bilinear')
        x7=self.fdm6(x6)
        x7+=x1
        x7=nn.functional.interpolate(x7, scale_factor=2, mode='bilinear')
        x8=self.fdm7(x7)
        x8+=x0
        x8=nn.functional.interpolate(x8, scale_factor=2, mode='bilinear')
        x9=self.fdm8(x8)
        x10=self.lasconv(x9)


        x5v=self.fdm4v(x4)
        x5v+=x3
        x5v=nn.functional.interpolate(x5v, scale_factor=7, mode='bilinear')
        x6v=self.fdm5(x5v)
        #print('x6v shape:',x6v.shape)
        #print('x2 shape:',x2.shape)
        x6v+=x2
        x6v=nn.functional.interpolate(x6v, scale_factor=2, mode='bilinear')
        x7v=self.fdm6(x6v)
        x7v=nn.functional.interpolate(x7v, scale_factor=2, mode='bilinear')
        x8v=self.fdm7(x7v)
        x8v+=x0
        x8v=nn.functional.interpolate(x8v, scale_factor=2, mode='bilinear')
        x9v=self.fdm8(x8v)
        x10Depth=self.lastconvd(x9v)
        return (x10, x10Depth)