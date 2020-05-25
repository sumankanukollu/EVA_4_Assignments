import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from IPython.core.debugger import set_trace

from torchvision.utils import make_grid
from torchsummary import summary
from PIL import Image
# New 
from torchvision import models
from zipfile import ZipFile
from pathlib import Path
import os ,sys, gc ,tarfile ,zipfile,pickle,pdb
from pprint import pprint

import argparse,json
parser = argparse.ArgumentParser()
parser.add_argument('-z', "--zipFName", help='Dataset Zip file name in full path', type=str)
parser.add_argument('-bs', "--batchSz", default=30,help='Batch size', type=int)
parser.add_argument('-e', "--epcs", help='Number of Epochs to run train and after test', type=int)
args = parser.parse_args()


from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
from zipfile import ZipFile

#%matplotlib inline
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc



class customDataset(Dataset):
    def __init__(self, zipFName,transfrm):
        self.z_obj      = ZipFile(zipFName)
        self.transfrm   = transfrm
        # BG Images : dataPath.parent.joinpath('bg')
        self.bg         = os.listdir('/content/drive/My Drive/EVA4/S15/dataset/bg')

        # bgfg
        tmp             = list(filter(lambda x : x.startswith('bg_fg_1/'),self.z_obj.namelist()))
        tmp.remove('bg_fg_1/') if 'bg_fg_1/' in tmp else tmp
        self.bgfgF      = tmp
        del tmp
        # masks
        tmp             = list(filter(lambda x : x.startswith('bg_fg_mask_1/'),self.z_obj.namelist()))
        tmp.remove('bg_fg_mask_1/') if 'bg_fg_mask_1/' in tmp else tmp
        self.maskF      = tmp
        del tmp
        #depth
        tmp             =   list(filter(lambda x : x.startswith('depthMap/'),self.z_obj.namelist()))
        tmp.remove('depthMap/') if 'depthMap/' in tmp else tmp
        self.depthF     = tmp
        del tmp

    def __len__(self):
        return len(self.bgfgF)

    def __getitem__(self, idx):
        bgname = os.path.basename(self.bgfgF[idx]).split('_bg_')[0]+'_bg.jpg'
        bgF    = os.path.join('/content/drive/My Drive/EVA4/S15/dataset/bg',bgname)

        bgImg   = self.transfrm(Image.open(bgF))
        bgfgImg = self.transfrm(Image.open((self.z_obj.open(self.bgfgF[idx]))))
        maskImg = self.transfrm(Image.open((self.z_obj.open(self.maskF[idx]))))
        depthImg = self.transfrm(Image.open((self.z_obj.open(self.depthF[idx]))).convert('L'))

        return {'bgK' : bgImg,'bgfgK': bgfgImg, 'maskK' : maskImg, 'depthK': depthImg}

    def __del__(self):
        self.z_obj.close()
        del self.bgfgF



def train(model,criterion,device,trainloader,optimizer,epochs,folderName=None):
    train_loss_mask, train_loss_depth,train_loss  = [],[],[]
    model.train()
    pbar = tqdm(trainloader)
    ### Train #####
    ################
    print('\n\n######################\nYou are in Train process\n################################')
    for batch_idx, data in enumerate(pbar):
        #print('Batch ID: ',batch_idx+1)
        gc.collect()
        optimizer.zero_grad()
        bg    = data['bgK'].to(device)
        bgfg  = data['bgfgK'].to(device)
        maskGt  = data['maskK'].to(device)
        depthGt = data['depthK'].to(device)
        ###########
        try:
            z=torch.cat([bgfg,bg], dim=1)
            (maskPred,depthPred) = model(z)  #(mask,depth)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print('| WARNING: ran out of memory, retrying batch',sys.stdout)
                print('Net parameters are : {}'.format(net.parameters()))
                #set_trace()
                sys.stdout.flush()
                for p in net.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                (maskPred,depthPred) = model(z)
            else:
                raise e
        ############
        #set_trace()
        loss_mask = criterion(maskPred,maskGt)
        train_loss_mask.append(loss_mask.item())
        loss_depth =criterion(depthPred,depthGt)
        train_loss_depth.append(loss_depth.item())
        loss = 2*loss_mask+loss_depth 
        train_loss.append(loss.item())
        loss_min = loss
        pbar.set_description(desc= "Overall Loss={:.4f} Mask loss={:.2f} Depth loss={:.2f}".format(loss.item(),loss_mask.item(), loss_depth.item()))
        loss.backward()
        optimizer.step()
        ###########
        ### Delete scope variables
        
        if batch_idx %50 == 0:
            print('batch_idx: {}'.format(batch_idx))
            print('loss: {}'.format(loss))
            print('##### Train Epoch: {}  [{}/{}  ({:.0f}%)]\tLoss:{:.6f}'.format(
                epochs,batch_idx*len(data),len(trainloader.dataset),
                100.*batch_idx/len(trainloader),loss.item()))
            #show(maskPred.detach().cpu())
            #show(depthPred.detach().cpu())
            #del maskPred,depthPred
    print('After compleation of training at epoch :{}'.format(batch_idx))
    #show(maskPred.detach().cpu())
    #show(depthPred.detach().cpu())
    if not os.path.exists(folderName.split('.zip')[0]):
        os.makedirs(folderName.split('.zip')[0])
    saveplot(maskPred.detach().cpu(),   os.path.join(folderName.split('.zip')[0],str(epoch)+'_train_'+'predicted_mask.jpg'))
    saveplot(maskGt.detach().cpu(),     os.path.join(folderName.split('.zip')[0],str(epoch)+'_train_'+'actual_mask.jpg'))
    saveplot(depthPred.detach().cpu(),  os.path.join(folderName.split('.zip')[0],str(epoch)+'_train_'+'predicted_depth.png'))
    saveplot(depthGt.detach().cpu(),    os.path.join(folderName.split('.zip')[0],str(epoch)+'_train_'+'actual_depth.png'))
    del maskPred,depthPred
    del bg, bgfg,maskGt,depthGt
    # create checkpoint variable and add important data
    checkpoint = {
        'epoch': epochs,
        'valid_loss_min': loss.item(),
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    return checkpoint,(train_loss_mask, train_loss_depth,train_loss)

        
def test(model,criterion,device,testloader,epochs,folderName=None):
    test_loss_mask, test_loss_depth,test_loss = [],[],[]
    model.eval()
    print('\n\n######################\nYou are in Test process\n################################')
    pbar = tqdm(testloader)
    for batch_idx, data in enumerate(pbar):
        with torch.no_grad():
            gc.collect()
            bg      = data['bgK'].to(device)
            bgfg    = data['bgfgK'].to(device)
            maskGt  = data['maskK'].to(device)
            depthGt = data['depthK'].to(device)
            ###########
            try:
                z=torch.cat([bgfg,bg], dim=1)
                (maskPred,depthPred) = model(z)  #(mask,depth)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print('| WARNING: ran out of memory, retrying batch',sys.stdout)
                    print('Net parameters are : {}'.format(net.parameters()))
                    #set_trace()
                    sys.stdout.flush()
                    for p in net.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    (maskPred,depthPred) = model(z)
                else:
                    raise e
            ############
            loss_mask = criterion(maskPred,maskGt).item()
            test_loss_mask.append(loss_mask)
            loss_depth =criterion(depthPred,depthGt).item()
            test_loss_depth.append(loss_depth)
            loss = 2*loss_mask+loss_depth 
            test_loss.append(loss)       
            if batch_idx %50 == 0:
                print('batch_idx: {}'.format(batch_idx))
                print('loss: {}'.format(loss))
                '''
                print('Test Epoch: {}  [{}/{}  ({:.0f}%)]\tLoss:{:.6f}'.format(
                    epochs,batch_idx*len(data),len(testloader.dataset),
                    100.*batch_idx/len(testloader),loss.item()))'''
                #show(maskPred.detach().cpu())
                #show(depthPred.detach().cpu())
                '''
                i = batch_idx
                print('inside save plot')
                
                saveplot(maskPred.detach().cpu(),'/content/drive/My Drive/EVA4/S15/modelWeights'+str(epoch)+'_'+str(i)+'_predicted_mask.jpg')
                saveplot(maskGt.detach().cpu(),'/content/drive/My Drive/EVA4/S15/modelWeights'+str(epoch)+'_'+str(i)+'_actual_mask.jpg')
                saveplot(depthPred.detach().cpu(),'/content/drive/My Drive/EVA4/S15/modelWeights'+str(epoch)+'_'+str(i)+'_predicted_depth.jpg')
                saveplot(depthGt.detach().cpu(),'/content/drive/My Drive/EVA4/S15/modelWeights'+str(epoch)+'_'+str(i)+'_actual_depth.jpg')
                '''
    #set_trace()
    print('\n####################\nTest set: Avg loss: {:.4f}, Mask Loss: {:.2f}, Depth Loss: {:.2f}\n############'.format(np.mean(test_loss), np.mean(test_loss_mask), np.mean(test_loss_depth)))
    print('After compleation of Test at epoch :{}'.format(batch_idx))
    
    if not os.path.exists(folderName.split('.zip')[0]):
        os.makedirs(folderName.split('.zip')[0])
    saveplot(bgfg.detach().cpu(),       os.path.join(folderName.split('.zip')[0],str(epoch)+'_test_'+'bgfg.jpg'))
    saveplot(maskPred.detach().cpu(),   os.path.join(folderName.split('.zip')[0],str(epoch)+'_test_'+'predicted_mask.jpg'))
    saveplot(maskGt.detach().cpu(),     os.path.join(folderName.split('.zip')[0],str(epoch)+'_test_'+'actual_mask.jpg'))
    saveplot(depthPred.detach().cpu(),  os.path.join(folderName.split('.zip')[0],str(epoch)+'_test_'+'predicted_depth.png'))
    saveplot(depthGt.detach().cpu(),    os.path.join(folderName.split('.zip')[0],str(epoch)+'_test_'+'actual_depth.png'))
    #show(maskPred.detach().cpu())
    #show(depthPred.detach().cpu())
    del bg, bgfg,maskGt,depthGt
    del maskPred,depthPred
    return (test_loss_mask, test_loss_depth,test_loss)

if __name__=='__main__':
    # 1. Declare paths
    homepath = r'/content/drive/My Drive/EVA4/S15'
    libPath  = r'/content/drive/My Drive/EVA4/S15/lib'
    utilsPath= r'/content/drive/My Drive/EVA4/S15/utils'
    scripts  = r'/content/drive/My Drive/EVA4/S15/scripts/pythonFiles'
    dataPath  = r'/content/drive/My Drive/EVA4/S15/dataset/zipFiles'

    sys.path.append(homepath)
    os.chdir(homepath)

    # 2. Identify device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('You are executing model on : %s'%device)
    
    # 2. Load transformations
    from utils.tfms import transformations 
    transform_train, transform_test = transformations()

    # 3. Initiate custom dataset
    from torch.utils.data import DataLoader,random_split
    print('Dataset using : {}'.format(args.zipFName))
    trainset = customDataset(zipFName = args.zipFName ,transfrm = transform_train)
    
    # 4. Split like 70% train and 30% test
    train_set, test_set  = random_split(trainset,[int(0.7*len(trainset)),int(0.3*len(trainset))])
    print('Train Dataset : {}\nTest Dataset : {}\n\t In batch size of : {}'.format(
        len(train_set)*3,len(test_set)*3,args.batchSz))

    #set_trace()
    # 5. Load dataloader
    trainloader = DataLoader(train_set, batch_size = args.batchSz, shuffle =True, pin_memory=True,num_workers =0)
    testloader  = DataLoader(test_set, batch_size = args.batchSz, shuffle =True, pin_memory=True,num_workers =0)

    # 6. display images
    from utils.img_save_show import show,saveplot
    #sample = next(iter(trainloader))
    #show(sample['bgK'])
    #show(sample['bgfgK'])
    #show(sample['maskK'])
    #show(sample['depthK'])
    #del sample 
    
    # 7 import models
    from scripts.pythonFiles.model import *
    net = DepthMask()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    from torchsummary import summary
    # Display Model Summary
    model = net.to(device)
    
    summary(model, input_size=(6, 224, 224))

    criterion = nn.BCEWithLogitsLoss() #SSIM( 3, reduction ="mean")
    optimizer = torch.optim.SGD(model.parameters(),lr= 0.01, momentum=0.9, weight_decay=1e-5)
    
    # 8. run train and test in epochs
    EPOCHS = args.epcs
    print('You are running {} dataset for {} epochs'.format(args.zipFName,args.epcs))
    
    # Prepare a dictionary 
    consol_loss = {}
    consol_loss['train_loss_mask']   = []
    consol_loss['train_loss_depth']  = []
    consol_loss['train_loss']        = []
    consol_loss['test_loss_mask']    = []
    consol_loss['test_loss_depth']   = []
    consol_loss['test_loss']         = []
    
    
    #set_trace()
    # Load dataset:
    #checkpoint_path  = r'/content/drive/My Drive/EVA4/S15/modelWeights/checkpoint/batch_best_ckp.pt'
    checkpoint_path  = r'/content/drive/My Drive/EVA4/S15/modelWeights/checkpoint/batch_best_ckp_gray.pt'
    if os.path.exists(checkpoint_path):
        print('Loading model : {}'.format('batch_best.ckp.pt'))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        loss = checkpoint['valid_loss_min']
        print('Model is loaded and valid_loss_min is :%s' %loss)

    #loss = 1e+4 # Initial loss
    for epoch in range(1,EPOCHS+1):
        print("\n###########EPOCH:%i#########\n"%epoch)
        checkpoint,(train_loss_mask, train_loss_depth,train_loss) = train(model, criterion, device, trainloader,optimizer, epochs=epoch,folderName=args.zipFName)
        if checkpoint['valid_loss_min'] < loss:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(loss,checkpoint['valid_loss_min']))
            # save checkpoint as best model
            #torch.save(checkpoint,r'/content/drive/My Drive/EVA4/S15/modelWeights/checkpoint/batch_best_ckp.pt')
            torch.save(checkpoint,r'/content/drive/My Drive/EVA4/S15/modelWeights/checkpoint/batch_best_ckp_gray.pt')
            print('\n\n\n**************Model saved**************\n\n\n')
            loss = checkpoint['valid_loss_min']
        (test_loss_mask, test_loss_depth,test_loss)    =  test(model,criterion,device,testloader,epoch,folderName=args.zipFName)
        
        consol_loss['train_loss_mask'].append(train_loss_mask)
        consol_loss['train_loss_depth'].append(train_loss_depth)
        consol_loss['train_loss'].append(train_loss)
        consol_loss['test_loss_mask'].append(test_loss_mask)
        consol_loss['test_loss_depth'].append(test_loss_depth)
        consol_loss['test_loss'].append(test_loss)

    with open(r'/content/drive/My Drive/EVA4/S15/modelWeights/consol_loss_pkl','ab') as f:
        pickle.dump(consol_loss,f)
        pickle.dump('\n\n',f)
        
    
    