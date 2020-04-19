import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        drpout = nn.Dropout(0.16)
        self.conv_layer1 = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # IP = 32  OP = 32  RF = 3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            drpout,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # IP = 32  OP = 32  RF = 5
            nn.BatchNorm2d(64),
            nn.ReLU(),
            drpout,
            nn.MaxPool2d(kernel_size=2, stride=2)                                # IP = 32  OP = 16  RF = 6
        )
        self.conv_layer2 = nn.Sequential(
            # Conv Layer block 2
            # Dilated conv
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2,dilation=2),  # IP = 16  OP = 16  RF = 14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            drpout,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,dilation=1), #IP= 16  OP= 16  RF=18
            nn.BatchNorm2d(128),
            nn.ReLU(),
            drpout,
            nn.MaxPool2d(kernel_size=2, stride=2),                                          # IP=16  OP= 8 RF= 20
            #nn.Dropout2d(p=0.05)
        )
        self.conv_layer3 = nn.Sequential(
            # Conv Layer block 3
            # Depthwise sep Conv
            #nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,groups=128),   # IP=8  OP=8  RF=28
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=1),                           #IP=8  OP=8   RF = 28   
            nn.BatchNorm2d(256),
            nn.ReLU(),
            drpout,
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),             # IP = 8   OP = 8  RF= 36
            nn.BatchNorm2d(128),
            nn.ReLU(),
            drpout,
            nn.MaxPool2d(kernel_size=2, stride=2)                                               # IP=8  OP=4  RF = 40
        )
        self.conv_layer4 = nn.Sequential(
            # Conv Layer block 4
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=1, padding=0),             # IP=4  OP=4  RF = 40
            nn.BatchNorm2d(10),
            nn.ReLU(),
            #drpout,
            #nn.Conv2d(in_channels=128, out_channels=10, kernel_size=1, padding=0),
            #nn.ReLU()
            nn.AvgPool2d(kernel_size=4)                                                       #IP=4  OP=1   RF=64    
          )

    def forward(self, x):
        # conv layers
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        #return x