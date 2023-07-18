import torch
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm
import pandas as pd
import monai
import os
from torchcontrib.optim import SWA
import torchvision.transforms as transforms
import numpy as np
from models import VIT_seg,pos_embed
from sklearn.metrics import accuracy_score
from dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from torch.optim.swa_utils import AveragedModel, SWALR
def compute_iou(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection

    return (intersection + 1e-15) / (union + 1e-15)

def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

gpus =[0]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

INPUT_SIZE =512

transform_test = transforms.Compose([
    transforms.Resize((512,512)),
])
data_path = '/Dataset/OriginalData'
val_set =Dataset(path=data_path, transform=transform_test, mode='test')
val_loader =DataLoader(val_set, batch_size =1, shuffle =False)
#pretrained model
model =torch.load("/checkpoints/net.pth", map_location='cpu').to('cuda:0')
print(model)
dices = 0
ious = 0
for index, (img, mask, name) in enumerate(val_loader):
    #output dir
    vis_path = "/log_results/"
    model.eval()
    with torch.no_grad():
        #input
        img =img.to('cuda:0')
        mask =mask.to('cuda:0')
        output =model(img)
        result = output[-1]
        result = torch.argmax(result, dim=1)
        result = result.reshape(result.shape[0],1,result.shape[1],result.shape[2]).type(dtype = torch.float32)
        gt_1 = img[0]
        gt_1 = gt_1[0].reshape(1,gt_1.shape[1],gt_1.shape[2])
        img_visualize =vutils.make_grid(result[0])
        iou =compute_iou(result[0],mask[0])
        ious =ious+iou
        masks = mask[0].reshape(512,512)
        results =result[0].reshape(512,512)
        masks = np.array(masks.cpu())
        results =np.array(results.cpu())
        dice = single_dice_coef(masks,results)
        dices =dices +dice
        img_visualize =vutils.make_grid(result[0])
        visualize_img_path = vis_path+str(name[0])
        vutils.save_image(img_visualize, visualize_img_path)

#this is iou not miou
print(ious / len(val_loader))
print(dices /len(val_loader))
