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

file_list =os.listdir("/organoid/Dataset/OriginalData/testing/images")
for files in file_list:
    name,_ = os.path.splitext(files)
    img_name = "/organoid/Dataset/OriginalData/testing/images/"+ files
    print(img_name)
    #img = Image.open(img_name).convert("L")
    img = cv2.imread(img_name,0)
    #img = transforms.ToTensor()(img).unsqueeze(0)
    #out,rs= model(img)
    #torch.save(rs,'/organoid/our_FFT/SR_results/test/{}.pt'.format(name))
    edges = cv2.Canny(img,100,255,apertureSize=3)
    cv2.imwrite('/organoid/our_FFT/edge_result/testing/{}.png'.format(name),edges)
    #print(rs.type())
    #print(rs.shape)






'''
img = Image.open('/home3/qinyiming/organoid/Dataset/OriginalData/training/images/0a5e4986-4b10-4ffd-bbdb-9fb11a194aec.png').convert('L')
ori = cv2.imread('/home3/qinyiming/organoid/Dataset/OriginalData/training/images/0a5e4986-4b10-4ffd-bbdb-9fb11a194aec.png',0)
img = transforms.ToTensor()(img).unsqueeze(0)
out,sr = model(img)
out = out.squeeze().detach().numpy()
sr = sr.squeeze().detach().numpy()
value_range =np.uint8(sr*255/(sr.max()-sr.min()))
out =np.uint8(out*255/(out.max()-out.min()))
# value_range =np.uint8(sr*255/(sr.max()-sr.min()))
# value_range[value_range>value_range.mean()] = 255
# value_range[value_range<value_range.mean()] = 0

# Canny Finding the edge
ori = cv2.blur(ori,(3,3))
edges = cv2.Canny(ori,100,255,apertureSize=3)
sr_edges = cv2.Canny(value_range,100,255,apertureSize=3)
th3 = cv2.adaptiveThreshold(ori,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
black = np.zeros_like(value_range)
# Hough circle transformation
#circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,20,
#                        param1=50,param2=30,minRadius=0,maxRadius=10000)

#for circle in circles[0]:
#    x = int(circle[0])
#    y = int(circle[1])
#    r = int(circle[2])
#    draw_circle = cv2.circle(black,(x,y) ,r ,(255,0,0) ,1,10 ,0) #画出检测到的圆，（255,255,255）代表白色

cv2.imwrite('sr0.png',np.concatenate([ori,out,value_range,th3,edges,sr_edges],axis=1))
'''