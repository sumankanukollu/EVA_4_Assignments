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