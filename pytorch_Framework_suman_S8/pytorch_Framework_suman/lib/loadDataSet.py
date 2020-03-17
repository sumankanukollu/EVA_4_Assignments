import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import augmentation
from utils import progress_bar

class dataSetFunctions:
    def __init__(self,root='./data',train=True,download = True):
        self.root 	= root
        self.train 	= train
        self.download = download
		
	def albumentationTransformations(self):
		transform_train = augmentation.AlbumentationTransformTrain()
		transform_test = augmentation.AlbumentationTransformTest()
		return (transform_train, transform_test)
		
		
    def transformation(self,type='train'):
        #import torchvision
        #from torchvision import datasets,transforms
        print('\n### Applying transformations for the type : {}'.format(type))
        if type== 'train':
        	return transforms.Compose(
        		[transforms.RandomCrop(32, padding=4),
        		transforms.RandomHorizontalFlip(),
        	    transforms.ToTensor(),
        		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        		]
        	)
        
        elif type== 'test':
        	return transforms.Compose(
        		[transforms.ToTensor(),
        		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        		]
        	)

    def dataSet(self,transform,name='cifar10'):
        print('\n### Preparing dataset for the name : {}'.format(name))
        if name == 'cifar10':
        	trainset = datasets.CIFAR10(
        		root 	= self.root,
        		train 	= True,
        		download= True,
        		transform= transform[0]
        	)
        	testset =  datasets.CIFAR10(
        		root 	= self.root,
        		train 	= False,
        		download= True,
        		transform= transform[1]
        	)   
        return (trainset,testset)
			
    def dataSetClasses(self,trainset):
        return trainset.classes
        	
    def dataLoader(self,trainDataset,testDataset):
        dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if torch.cuda.is_available() else dict(shuffle=True, batch_size=64)
        print('\n### Loading data from dataset')
        # train dataloader
        train_loader = torch.utils.data.DataLoader(trainDataset, **dataloader_args)
        
        # test dataloader
        test_loader = torch.utils.data.DataLoader(testDataset, **dataloader_args)
        return (train_loader,test_loader)
        
    def datasetMeanVarStd(self,dset):
        import numpy as np
        #trainDataset = self.dataSet(type=type)
        print('\n### Type of the given dataset is :{}'.format(type(dset.data)))
        print('\n### Shape of the given dataset is : {}'.format(dset.data.shape))
        print('\n### Mean is : {}'.format(np.mean(dset.data)))
        print('\n### Var is  : {}'.format(np.var(dset.data)))
        print('\n### Std is : {}'.format(np.std(dset.data)))
