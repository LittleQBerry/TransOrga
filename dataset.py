import os
import torch
from PIL import Image
from torchvision import transforms
from torch import nn
import numpy as np
import cv2
class Dataset(torch.utils.data.Dataset):
    '''
    train, test, validatiaon
    '''
    def __init__(self, path, transform=None, mode='train', val_path = None):
        self.img_path = path
        self.mask_path = path
        self.transform = transform
        self.mode = mode
        if self.mode == 'train':
            self.img_path =self.img_path +"/training/images/"
            self.mask_path = self.mask_path +"/training/segmentations/"
            self.edge_path ='/edge_result/training/'
            self.sr_path = '/SR_results/training/'
            train_file = os.listdir(self.img_path)
            files=train_file
        elif self.mode == 'validation':
            self.img_path =self.img_path +"/validation/images/"
            self.mask_path = self.mask_path +"/validation/segmentations/"
            self.edge_path = '/edge_result/validation/'
            self.sr_path ='/SR_results/validation/'
            valid_file = os.listdir(self.img_path)
            files =valid_file
        elif self.mode == 'test':
            self.img_path =self.img_path +"/testing/images/"
            self.mask_path = self.mask_path +"/testing/segmentations/"
            self.edge_path = '/edge_result/testing/'
            self.sr_path = '/SR_results/testing/'
            test_file = os.listdir(self.img_path)
            files=test_file
        self.files = files
        #self.log = log

    

    def __getitem__(self,index):
        if self.mode == 'train' :
            name,_ = os.path.splitext(self.files[index])
            img = Image.open(self.img_path+self.files[index].strip()).convert('L')
            mask = Image.open(self.mask_path+self.files[index].strip()).convert('L')
            edge = Image.open(self.edge_path+self.files[index].strip()).convert('L')
            seed = torch.random.seed()
            sr =torch.load("{}/{}.pt".format(self.sr_path,name))
            sr =sr[0]          
            torch.random.manual_seed(seed)
            img = self.transform(img)
            sr =self.transform(sr)
            #sr = transforms.ToTensor()(sr)
            img = transforms.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5)(img)
            img = transforms.ToTensor()(img)
            torch.random.manual_seed(seed)
            mask = self.transform(mask)
            mask =transforms.ToTensor()(mask)
            mask = torch.where(mask>0.0,1.0,0.0)

            edge = self.transform(edge)
            edge = transforms.ToTensor()(edge)


            file_name = self.files[index].strip()
            img_r =torch.cat([img,sr],dim=0)
            return img_r, mask, file_name
        elif self.mode == 'validation':
            
            name,_ = os.path.splitext(self.files[index])
            img = Image.open(self.img_path+self.files[index].strip()).convert('L')
            mask = Image.open(self.mask_path+self.files[index].strip()).convert('L')
            edge = Image.open(self.edge_path+self.files[index].strip()).convert('L')
            seed = torch.random.seed()
            sr =torch.load("{}/{}.pt".format(self.sr_path,name))
            sr =sr[0]
            
            torch.random.manual_seed(seed)
            img = self.transform(img)
            sr = self.transform(sr)
            img = transforms.ToTensor()(img)
            torch.random.manual_seed(seed)
            mask = self.transform(mask)
            mask =transforms.ToTensor()(mask)
            mask = torch.where(mask>0.0,1.0,0.0)

            edge = self.transform(edge)
            edge =transforms.ToTensor()(edge)
            file_name = self.files[index].strip()
            img_r =torch.cat([img,sr],dim=0)
            return img_r, mask, file_name
        elif self.mode == 'test':
            name,_ = os.path.splitext(self.files[index])
            img =cv2.imread(self.img_path+self.files[index].strip(),0)
            img =Image.fromarray(img.astype('uint8'))
            mask = Image.open(self.mask_path+self.files[index].strip()).convert('L')
            edge = Image.open(self.edge_path+self.files[index].strip()).convert('L')
            seed = torch.random.seed()
            sr =torch.load("{}/{}.pt".format(self.sr_path,name))
            sr =sr[0]
            
            torch.random.manual_seed(seed)
            img = self.transform(img)
            sr = self.transform(sr)
            img = transforms.ToTensor()(img)
            torch.random.manual_seed(seed)
            mask = self.transform(mask)
            mask =transforms.ToTensor()(mask)
            mask = torch.where(mask>0.0,1.0,0.0)
            print(img.shape)
            edge = self.transform(edge)
            edge =transforms.ToTensor()(edge)
            file_name = self.files[index].strip()
            img_r =torch.cat([img,sr],dim=0)
            return img_r, mask, file_name

    def __len__(self):
        return len(self.files)
