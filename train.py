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

from utils import DiceLoss, get_compactness_cost, FocalLoss,FocalLoss_Binary

def train_epoch(net, train_iter, test_iter, loss,lossd, optimizer, start_epoch,num_epochs, scheduler, devices=d2l.try_all_gpus(), weight_path = None, data_dir = None):
    timer, num_batches = d2l.Timer(), len(train_iter)

    print(devices)
    #log info
    loss_list = []
    cla_loss_list = []
    train_acc_list = []
    train_cla_acc_list = []
    test_acc_list = []
    test_cla_acc_list = []
    epochs_list = []
    time_list = []
    lr_list = []
    
 
    for epoch in range(start_epoch,start_epoch+num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(7)
        test_metric =d2l.Accumulator(2)
        cla_preds = []
        cla_gt = []
        for i, (X, labels,file_name) in enumerate(train_iter):
            print(f"epoch:{epoch}:{i}/{len(train_iter)}")
            timer.start()
            if isinstance(X, list):
                X = [x.cuda(non_blocking =True) for x in X]
            else:
                X = X.cuda(non_blocking =True)
            gt = labels.long().cuda(non_blocking =True).squeeze()
            net.train()
            optimizer.zero_grad()
            #mutli-level output
            result =net(X)

            input_gt = torch.nn.functional.one_hot(gt, 2).permute(0,3,1,2).type(dtype = torch.float32)
            seg_loss = loss(result[-1], input_gt)
            aux_loss_1 = loss(result[0], input_gt)
            aux_loss_2 = loss(result[1], input_gt)
            aux_loss_3 = loss(result[2], input_gt)

            loss_dice = lossd(result[-1],gt, softmax=True)
            output = torch.argmax(result[-1], dim=1)
            output = output.reshape(output.shape[0],1,output.shape[1],output.shape[2]).type(dtype = torch.float32)
            loss_com = get_compactness_cost(output)

            loss_sum = (seg_loss +0.2*aux_loss_1 + 0.3*aux_loss_2 + 0.4*aux_loss_3)+0.5*loss_dice+0.2*loss_com
            l = loss_sum
            loss_sum.sum().backward()
            optimizer.step()

            acc = d2l.accuracy(result[-1], gt)
            metric.add(l, acc, labels.shape[0], labels.numel(),loss_com,loss_dice,seg_loss)
            timer.stop()
        
        
        scheduler.step()
        
 
        
        
        for index,(img, mask,file_name) in enumerate(test_iter):
            vis_path = './log/vis/'
            net.eval()
            img =img.cuda(non_blocking =True)
            mask =mask.cuda(non_blocking =True)
            gt = mask.long().cuda(non_blocking =True).squeeze()
            with torch.no_grad():
                #multi level results
                output = net(img)
                result = output[-1]
                test_acc = d2l.accuracy(result, gt)
                test_metric.add(test_acc, mask.numel())
                #print(result.shape)
                result = torch.argmax(result, dim=1)
                result = result.reshape(result.shape[0],1,result.shape[1],result.shape[2]).type(dtype = torch.float32)
                gt_1 = img[0]
                gt_1 = gt_1[0].reshape(1,gt_1.shape[1],gt_1.shape[2])
                
                if index%5 ==0:
                    img_list =[
                        #img[0,:,:,:],
                        gt_1,
                        result[0,:,:,:],
                        mask[0,:,:,:]
                    ]
                    img_visualize = vutils.make_grid(img_list)
                    visualize_img_path = vis_path+str(epoch)+'_'+str(index+1)+'.png'
                    vutils.save_image(img_visualize, visualize_img_path)
        
        print(f"epoch {epoch+1}/{start_epoch+epochs_num} --- loss {metric[0] / metric[2]:.3f} \
            --- train seg acc {metric[1] / metric[3]:.3f}  --- test seg acc {test_metric[0] /test_metric[1]:.3f}\
            --- lr {optimizer.state_dict()['param_groups'][0]['lr']} --- cost time {timer.sum()}\
             --- com_loss {metric[4] / metric[2]:.3f} --- dice_loss {metric[5] / metric[2]:.3f} --- seg_loss {metric[6] / metric[2]:.3f}")
        #---------output------------
        df = pd.DataFrame()
        loss_list.append((metric[0]+metric[4]+metric[5]+metric[6]) / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_metric[0] /test_metric[1] )
        epochs_list.append(epoch+1)
        time_list.append(timer.sum())
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        
        df['epoch'] = epochs_list
        df['loss'] = loss_list
        df['train_seg_acc'] = train_acc_list
        df['test_seg_acc'] = test_acc_list
        df["lr"] = lr_list
        df['time'] = time_list
        
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        df.to_csv(os.path.join(data_dir,"test.csv"),index=False)
        #----------------save model------------------- 
        if epoch % 50 == 0:
            torch.save(net,os.path.join(weight_path,f'net_{epoch+1}.pth'))
 
    #save final
    torch.save(net, os.path.join(weight_path,f'net_{epoch+1}.pth'))


#torch home
os.environ['TORCH_HOME'] ='/.cache/'

CUDA_LAUNCH_BLOCKING=1
gpus =[0]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))



INPUT_SIZE= 512

MEAN = [108.64628601 / 255, 75.86886597 / 255, 54.34005737 / 255]
STD = [70.53946096 / 255, 51.71475228 / 255, 43.03428563 / 255]
transform_train = transforms.Compose([
    transforms.Resize((INPUT_SIZE,INPUT_SIZE)),
])
transform_test = transforms.Compose([
    transforms.Resize((INPUT_SIZE,INPUT_SIZE)),  
])

#dataset 
batch_size = 4
val_batch_size = 1
data_path = '/Dataset/OriginalData'
train_set = Dataset(path=data_path, transform =transform_train, mode ='train')
val_set =Dataset(path=data_path, transform=transform_test, mode='validation')

train_loader =DataLoader(train_set,batch_size =batch_size, shuffle =True)
val_loader =DataLoader(val_set, batch_size =val_batch_size, shuffle =False)


## model initial
model = VIT_seg.vit_base_patch16(img_size=INPUT_SIZE, weight_init="nlhb", seg_num_classes=2, num_classes=2, out_indices = [2, 5, 8, 11])


checkpoint_model = torch.load('./vit_base.pth')['model']
state_dict = model.state_dict()

for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# interpolate position embedding
pos_embed.interpolate_pos_embed(model, checkpoint_model)

# distributed training
model.load_state_dict(checkpoint_model, strict=False)
model = nn.DataParallel(model.to('cuda:0'), device_ids=gpus, output_device=gpus[0])

##

#load pre-trained model
#model =torch.load("/organoid/our_FFT/log/checkpoints/net_121.pth", map_location='cpu').to('cuda:0')

#print(model)
# training setting
start_epoch =0
epochs_num = 400
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
schedule =torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)

lossd =DiceLoss(2)
lossf =FocalLoss_Binary()
train_epoch(model, train_loader, val_loader, lossf, lossd,optimizer, start_epoch,epochs_num, scheduler=schedule, weight_path="./log/checkpoints", data_dir="./log/log")
