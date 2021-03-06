# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import torch
import torchvision
import zipfile
import requests
from io import StringIO, BytesIO
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
import albumentations as alb
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
import io,glob,os,time,random
from shutil import move
from os.path import join
from os import listdir,rmdir
import torch.utils.data as data
from torchvision.datasets import ImageFolder

# https://albumentations.readthedocs.io/en/latest/_modules/albumentations/augmentations/transforms.html

class album_compose_train:
    def __init__(self):
        #meandata, stddata=album_calculate_dataset_mean_std()
        mean_color=(0.4804, 0.4482, 0.3976)
        #channel_mean=(meandata[0]+meandata[1]+meandata[2])/3.0
        #print('channel mean',channel_mean)
        self.albtransform  = self.transform = alb.Compose([
                                alb.PadIfNeeded(value=[4,4,4]),
                                alb.RandomCrop(56,56, p=1.),
                                alb.HorizontalFlip(p=0.25),
                                alb.Rotate(limit=10),
                                alb.ShiftScaleRotate(),
                                alb.GaussNoise(),
                                alb.GlassBlur(),
                                alb.CoarseDropout(max_holes=1, max_height=32, max_width=32, min_height=8, min_width=8, fill_value=0.4421*255.0, 
                                    p=0.5),
                                alb.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=0.4421*255, always_apply=False, p=0.5),
                                alb.Normalize((0.4804, 0.4482, 0.3976), (0.227, 0.269, 0.282)),
                                ToTensor()
        ])
        self.albtransform_suman = alb.Compose([
                                #alb.PadIfNeeded(min_height=56, min_width=56, border_mode=cv2.BORDER_REFLECT_101, value=np.array(mean_color)*255),
                                alb.RandomCrop(56,56,always_apply=True),
                                alb.ShiftScaleRotate(scale_limit = (0.9,1.08)),
                                alb.RandomContrast(limit = (0.7,1.3)),
                                alb.HorizontalFlip(p=0.5),
                                
                                ##alb.Rotate((-25.0,25.0)),
                                #alb.ToGray(),
                                #alb.RandomBrightnessContrast(),
                                ##alb.RandomResizedCrop(64, 64, scale=(0.75, 1.0), ratio=(0.9, 1.1), p=0.75),
                                
                                alb.RandomGamma(),
                                alb.Cutout(num_holes=2,max_h_size=16, max_w_size=16,fill_value=0.4421*255),
                                alb.Normalize((0.4804, 0.4482, 0.3976), (0.277, 0.269, 0.282)),
                                ToTensor(),
                                
        ])
     
        '''#meandata, stddata=album_calculate_dataset_mean_std()

        #channel_mean=(meandata[0]+meandata[1]+meandata[2])/3.0
        #print('channel mean',channel_mean)
        self.albtransform = alb.Compose([
        
        alb.PadIfNeeded(value=[4,4,4]),
        alb.RandomCrop(64,64),
        alb.Rotate(limit=10),
        alb.ShiftScaleRotate(),
        alb.HorizontalFlip(p=0.5),
        #alb.Cutout(num_holes=1,max_h_size=16, max_w_size=16,fill_value=127.5),
        alb.Cutout(num_holes=1,max_h_size=16, max_w_size=16),
        alb.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTensor(),
        ])'''
        
    def __call__(self,img):
        img=np.array(img)
        img=self.albtransform(image=img)
        return img['image']

class album_compose_test:
  def __init__(self):
        #meandata, stddata=album_calculate_dataset_mean_std()
        self.albtransform = alb.Compose([
        
        #print('meandata[0]:',meandata[0]),
        alb.Normalize((0.4804, 0.4482, 0.3976), (0.277, 0.269, 0.282)),
        ToTensor(),
        ])
 
  def __call__(self,img):
    img=np.array(img)
    img=self.albtransform(image=img)
    return img['image']


def download_images(url,homepath='/content/drive/My Drive/pytorch_Framework_suman/'):
    if os.path.exists(os.path.join(homepath,'data/tinyImagenet/tiny-imagenet-200.zip')):
        #if not os.path.exists(os.path.join(homepath,'data/tinyImagenet/tiny-imagenet-200.zip')):
        print('Images already downloaded...')
        return
    else:
        r= requests.get(url, stream=True)
        print("downloading " +url)
        zip_ref=zipfile.ZipFile(BytesIO(r.content))
        zip_ref.extractall('./')
        print('download is success')
        zip_ref.close()

	
def copyValDsIntoTrain():
    os.chdir('/content/')

    target_folder = './tiny-imagenet-200/val/'
    dest_folder = './tiny-imagenet-200/train/'

    val_dict = {}

    with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            splitline = line.split('\t')
            val_dict[splitline[0]] = splitline[1]
        paths = glob.glob('./tiny-imagenet-200/val/images/*')

        for path in paths:
            file = path.split('/')[-1].split('\\')[-1]
            folder = val_dict[file]
            dest = dest_folder + str(folder) + '/images/' + str(file)
            move(path, dest)



def splitDs():
    os.chdir('/content/')
    target_folder = './tiny-imagenet-200/train/'
    train_folder = './tiny-imagenet-200/train_set/'
    test_folder = './tiny-imagenet-200/test_set/'

    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)

    paths = glob.glob('./tiny-imagenet-200/train/*')

    for path in paths:
        folder = path.split('/')[-1].split('\\')[-1]
        source = target_folder + str(folder + '/images/')
        train_dest = train_folder + str(folder + '/')
        test_dest = test_folder + str(folder + '/')
        if not os.path.exists(train_dest):
            os.mkdir(train_dest)
        if not os.path.exists(test_dest):
            os.mkdir(test_dest)
        images = glob.glob(source + str('*'))
        # print(len(images))
        # making random
        random.shuffle(images)

        test_imgs = images[:165].copy()
        train_imgs = images[165:].copy()

        # moving 30% for validation
        for image in test_imgs:
            file = image.split('/')[-1].split('\\')[-1]
            dest = test_dest + str(file)
            move(image, dest)

        # moving 70% for training
        for image in train_imgs:
            file = image.split('/')[-1].split('\\')[-1]
            dest = train_dest + str(file)
            move(image, dest)
    print('test_set dir len is : {}'.format(len(os.listdir('./tiny-imagenet-200/test_set/n01629819/'))))
    print('train_set dir len is : {}'.format(len(os.listdir('./tiny-imagenet-200/train_set/n01629819/'))))

def load_data():
    download_images('http://cs231n.stanford.edu/tiny-imagenet-200.zip')
    copyValDsIntoTrain()
    splitDs()

    os.chdir('/content/tiny-imagenet-200')
    train_ds = ImageFolder('train_set', transform=album_compose_train())
    test_ds = ImageFolder('test_set', transform=album_compose_test())
    SEED = 1

    cuda = torch.cuda.is_available()
    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=1024, num_workers=8, pin_memory=True) if cuda else dict(shuffle=True, batch_size=128)

    # train dataloader
    train_dl = torch.utils.data.DataLoader(train_ds, **dataloader_args)

    # test dataloader
    test_dl = torch.utils.data.DataLoader(test_ds, **dataloader_args)


    return train_dl, test_dl, train_ds, test_ds


def album_calculate_dataset_mean_std():

    trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=ToTensor())
    testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=ToTensor())

    data = np.concatenate([trainset.data, testset.data], axis=0)
    data = data.astype(np.float32)/255.

    print("\nTotal dataset(train+test) shape: ", data.shape)

    means = []
    stdevs = []

    for i in range(3): # 3 channels
        pixels = data[:,:,:,i].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    return [means[0], means[1], means[2]], [stdevs[0], stdevs[1], stdevs[2]]