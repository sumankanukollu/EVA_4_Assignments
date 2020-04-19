import io,glob,os,time,random
from shutil import move
from os.path import join
from os import listdir,rmdir

import scipy.ndimage as nd
import numpy as np


def copyValDsIntoTrain():
    os.chdir('/content/')
    target_folder = './tiny-imagenet-200/val/'
    dest_folder = './tiny-imagenet-200/train/'
    
    val_dict={}
    
    with open('./tiny-imagenet-200/val/val_annotations.txt','r') as f:
        for line in f.readlines():
            splitline = line.split('\t')
            val_dict[splitline[0]] = splitline[1]
        paths = glob.glob('./tiny-imagenet-200/val/images/*')
      
        for path in paths:
            file = path.split('/')[-1].split('\\')[-1]
            folder = val_dict[file]
            dest = dest_folder + str(folder) + '/images/' + str(file)
            move(path,dest)
    		
def splitDs():
    # Here 70% and 30%
    target_folder = './tiny-imagenet-200/train/'
    train_folder = './tiny-imagenet-200/train_set/'
    test_folder = './tiny-imagenet-200/test_set/'
     
    if not os.path.exists(train_folder) and not os.path.exists(test_folder):
        os.mkdir(train_folder)
        os.mkdir(test_folder)
     
        paths = glob.glob('./tiny-imagenet-200/train/*')
        
        for path in paths:
            folder = path.split('/')[-1].split('\\')[-1]
            source = target_folder + str(folder + '/images/')
            train_dest = train_folder + str(folder + '/')
            test_dest = test_folder + str(folder + '/')
    
            os.mkdir(train_dest)
            os.mkdir(test_dest)
            images = glob.glob(source + str('*'))
            #print(len(images))
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