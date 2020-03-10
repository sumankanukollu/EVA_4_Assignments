class dataSetClass:
    def __init__(self,root='./root',train=True,download = True):
        self.root 	= root
        self.train 	= train
        self.download = download
    def transformation(self,type='train'):
        import torchvision
        from torchvision import datasets,transforms
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
    	
    def dataSet(self,name='cifar10',type='train'):
        import torchvision
        from torchvision import datasets,transforms
        print('\n### Preparing dataset for the name : {} and type : {}'.format(name,type))
        if name == 'cifar10' and type == 'train':
        	return datasets.CIFAR10(
        		root 	= './root',
        		train 	= True,
        		download= True,
        		transform= self.transformation(type=type)
        	)
        elif name == 'cifar10' and type == 'test':
        	return datasets.CIFAR10(
        		root 	= './root',
        		train 	= False,
        		download= True,
        		transform= self.transformation(type=type)
        	)   
    def dataSetClasses(self):
        return self.dataSet().classes
        	
    def dataLoader(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        SEED = 1
        # For reproducibility
        torch.manual_seed(SEED)
        
        # CUDA?
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)
        
        
        if cuda:
            torch.cuda.manual_seed(SEED)
        
        # dataloader arguments - something you'll fetch these from cmdprmt
        dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
        print('\n### Loading data from dataset')
        # train dataloader
        train_loader = torch.utils.data.DataLoader(self.dataSet(type='train'), **dataloader_args)
        
        # test dataloader
        test_loader = torch.utils.data.DataLoader(self.dataSet(type='test'), **dataloader_args)
        return (train_loader,test_loader)
        
    def datasetMeanVarStd(self,dset):
        import numpy as np
        #trainDataset = self.dataSet(type=type)
        print('\n### Type of the given dataset is :{}'.format(type(dset.data)))
        print('\n### Shape of the given dataset is : {}'.format(dset.data.shape))
        print('\n### Mean is : {}'.format(np.mean(dset.data)))
        print('\n### Var is  : {}'.format(np.var(dset.data)))
        print('\n### Std is : {}'.format(np.std(dset.data)))
 
