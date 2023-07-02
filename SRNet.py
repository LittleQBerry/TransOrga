import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from math import sqrt
import numpy  as np
from PIL import Image
from torchvision import transforms
import cv2
import os
class SRlayer_(nn.Module):
    def __init__(self,channel):
        super(SRlayer_,self).__init__()
        self.channel = channel
        self.batch = 1
        self.output_conv = nn.Conv2d(2,1, kernel_size=1)
        self.bn  = nn.BatchNorm2d(1)
        self.Relu = nn.ReLU()
        
        nn.init.kaiming_normal_(self.output_conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.kernalsize = 3
        self.amp_conv = nn.Conv2d(self.channel, self.channel, kernel_size=self.kernalsize, stride=1,padding=1, bias=False)
        self.fucker = np.zeros([self.kernalsize,self.kernalsize])
        for i in range(self.kernalsize):
            for j in range(self.kernalsize):
                self.fucker[i][j] = 1/np.square (self.kernalsize)
        self.aveKernal = torch.Tensor(self.fucker).unsqueeze(0).unsqueeze(0).repeat(self.batch,self.channel,1,1)
        self.amp_conv.weight = nn.Parameter(self.aveKernal, requires_grad=False)
        
        self.amp_relu = nn.ReLU()
        self.gaussi = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1,padding=1, bias=False)
        self.gauKernal = torch.Tensor([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]).unsqueeze(0).unsqueeze(0).repeat(self.batch,self.channel,1,1)
        self.gaussi.weight = nn.Parameter(self.gauKernal,requires_grad=False)
 
    def forward(self,x):
        out = []
        SRs = []
        for batch in range(x.shape[0]):
            x1 = x[batch,0,:,:].unsqueeze(0).unsqueeze(0)
            rfft = torch.fft.fftn(x1)
            amp = torch.abs(rfft) + torch.exp(torch.tensor(-10))
            log_amp = torch.log(amp)
            phase = torch.angle(rfft)
            amp_filter = self.amp_conv(log_amp)
            amp_sr = log_amp - amp_filter
            SR = torch.fft.ifftn(torch.exp(amp_sr+1j*phase))
            SR = torch.abs(SR)
            SR = self.gaussi(SR)
            y = torch.cat([SR,x1],dim=1)
            y = self.output_conv(y)
            y = self.bn(y)
            y = self.Relu(y)
            SRs.append(SR)

            out.append(y)
        
        return torch.cat(out,dim=0),torch.cat(SRs,dim=0)

model  = SRlayer_(1)
#please replace the dirs whit your owns.
file_list =os.listdir("/Dataset/OriginalData/testing/images")
for files in file_list:
    name,_ = os.path.splitext(files)
    #original dir (change)
    img_name = "/Dataset/OriginalData/testing/images/"+ files
    print(img_name)
    #for SR
    img = Image.open(img_name).convert("L")
    
    img = transforms.ToTensor()(img).unsqueeze(0)
    
    out,rs= model(img)
    #SR output dir (change)
    torch.save(rs,'/SR_results/testing/{}.pt'.format(name))
    #for edge
    #img = cv2.imread(img_name,0)
    #edge dir (change)
    #edges = cv2.Canny(img,100,255,apertureSize=3)
    #cv2.imwrite('/edge_result/testing/{}.png'.format(name),edges)
    
