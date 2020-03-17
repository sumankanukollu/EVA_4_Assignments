import os
import PIL
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import augmentation
#from utils import progress_bar
from torchvision.utils import make_grid, save_image

class utils_showimgs:
    def __init__(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import torchvision

    def imshow(self,img):
        import matplotlib.pyplot as plt
        import numpy as np
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def getRandomTrainImgs(self,trainDataset,train_loader):
        import matplotlib.pyplot as plt
        import numpy as np
        import torchvision
        # get some random training images
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        print(len(images))
        print(trainDataset.class_to_idx)
        print(labels)

        # show images
        self.imshow(torchvision.utils.make_grid(images[:4]))
        classes = trainDataset.classes
        # print labels
        print(' '.join('%10s' % classes[labels[j]] for j in range(4)))
        
    
    def loadImage(self,imagedirectory, imagename):
        # Load Image
        imgpath = os.path.join(imagedirectory, imagename)
        pilimg = PIL.Image.open(imgpath)
        return pilimg
        
    def saveimage(self,images, outputdirectory, imagename):
        os.makedirs(outputdirectory, exist_ok=True)
        outputname = imagename
        outputpath = os.path.join(outputdirectory, outputname)
        save_image(images, outputpath)
        pilimg = PIL.Image.open(outputpath)
        return pilimg
        
    