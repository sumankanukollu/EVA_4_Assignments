'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

######### Session-11 Model on CIFAR-10  START

import torch.nn as nn
import torch.nn.functional as F

def conv_2d(ni, nf, ks=3, s=1, p=1,b=False, padding_mode='zeros'):
    return nn.Conv2d(in_channels=ni, out_channels=nf, kernel_size=ks, stride=s, padding=p,bias=False, padding_mode='zeros')

def conv_bn_relu(ni, nf,ks,s,p):
    return nn.Sequential(conv_2d(ni, nf,ks,s,p),
                         nn.BatchNorm2d(nf), 
                         nn.ReLU(inplace=True), 
                         )
    
def resBlock(ni,nf,ks,s,p):
    return nn.Sequential(conv_bn_relu(ni, nf,ks,s,p),
                         conv_bn_relu(ni, nf,ks,s,p) 
                         )
    
class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x.view(x.size(0), -1)

    

    
class s11Model(nn.Module):
    def __init__(self):
        super(s11Model, self).__init__()
        self.prepLayer = conv_bn_relu(ni=3,nf=64,ks=3,s=1,p=1)
        ########Layer-1
        self.layer1x   = nn.Sequential(conv_2d(ni=64,nf=128,ks=3,s=1,p=1),
                                       nn.MaxPool2d(2,2),
                                       nn.BatchNorm2d(128), 
                                      nn.ReLU(inplace=True)
        )
        self.R1       = resBlock(ni=128, nf=128,ks=3,s=1,p=1)
        ########Layer-2
        self.Layer2   = nn.Sequential(conv_2d(ni=128,nf=256,ks=3,s=1,p=1),
                                      nn.MaxPool2d(2,2),
                                      nn.BatchNorm2d(256), 
                                      nn.ReLU(inplace=True)
        )
        ####### Layer-3
        self.layer3x   = nn.Sequential(conv_2d(ni=256,nf=512,ks=3,s=1,p=1),
                                       nn.MaxPool2d(2,2),
                                       nn.BatchNorm2d(512), 
                                      nn.ReLU(inplace=True)
        )
        self.R2       = resBlock(ni=512, nf=512,ks=3,s=1,p=1)
        ####### MaxPooling with Kernel Size 4
        self.mp4      = nn.MaxPool2d(4,4)
        ####### FC Layer
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prepLayer(x)
        x = self.layer1x(x)
        r1 = self.R1(x)
       
        #x  = x + r1
        x  = x.add(r1)
        x = self.Layer2(x)
        x = self.layer3x(x)
        r2 = self.R2(x)
       
        #x  = x + r2
        x = x.add(r2)
        x = self.mp4(x)
        #print('size of x befor linear : {}'.format(x.shape))
        ## Flatten 
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        #print('size of x is After linear : {}'.format(x.shape))
        x = F.log_softmax(x, dim=-1)
        #print('size of x is After softmax : {}'.format(x.shape))
        return x
        
######### Session-11 Model on CIFAR-10  END