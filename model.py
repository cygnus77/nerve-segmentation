#pylint: disable=E1101
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    num_classes = 1
    
    def __init__(self):
        super(Net, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.encoder = vgg16.features
        for i,param in enumerate(self.encoder.parameters()):
            param.requires_grad = i >= 16
        self.decoder1x1 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.convT2d1 = nn.ConvTranspose2d(in_channels=1, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.convT2d2 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, stride=4, padding=0)
        self.convT2d3 = nn.ConvTranspose2d(in_channels=512, out_channels=1, kernel_size=4, stride=4, padding=0)
    
    def forward(self, x):

        skipConnections = {}
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [23, 15]:
                skipConnections[i] = x
                
        x = self.decoder1x1(x)
        x = self.convT2d1(x)
        x = torch.cat((x,skipConnections[23]), 1)
        x = self.convT2d2(x)
        x = torch.cat((x, skipConnections[15]), 1)
        x = self.convT2d3(x)
        x = nn.Sigmoid()(x)
        x = x.view(x.size()[0], -1, Net.num_classes)
        
        return x

