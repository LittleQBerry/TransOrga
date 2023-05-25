import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from math import sqrt
import numpy  as np
from PIL import Image
from torchvision import transforms
import cv2
class SRlayer_(nn.Module):
    def __init__(self,channel):
        super(SRlayer_,self).__init__()
        self.channel = channel
        self.batch = 1
        self.output_conv = nn.Conv2d(2,1, kernel_size=1)
        self.bn  = nn.BatchNorm2d(1)
        self.Relu = nn.ReLU()
        
        nn.init.kaiming_normal_(self.output_conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.kernalsize = 3
        self.amp_conv = nn.Conv2d(self.channel, self.channel, kernel_size=self.kernalsize, stride=1,padding=1, bias=False)
        self.fucker = np.zeros([self.kernalsize,self.kernalsize])
        for i in range(self.kernalsize):
            for j in range(self.kernalsize):
                self.fucker[i][j] = 1/np.square(self.kernalsize)
        self.aveKernal = torch.Tensor(self.fucker).unsqueeze(0).unsqueeze(0).repeat(self.batch,self.channel,1,1)
        self.amp_conv.weight = nn.Parameter(self.aveKernal, requires_grad=False)
        
        self.amp_relu = nn.ReLU()
        self.gaussi = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1,padding=1, bias=False)
        self.gauKernal = torch.Tensor([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]).unsqueeze(0).unsqueeze(0).repeat(self.batch,self.channel,1,1)
        self.gaussi.weight = nn.Parameter(self.gauKernal,requires_grad=False)

        # for m in self.children():
        #     if isinstance(m, nn.Linear):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, -100)
        #     # 也可以判断是否为conv2d，使用相应的初始化方式 
        #     elif isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)  