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
        
    def plotmisclassifiedimages(self,model, device, classes, testloader, numofimages = 25, savefilename="misclassified"):
        model.eval()
        misclassifiedcounter = 0
        fig = plt.figure(figsize=(10,9))
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)     # get the index of the max log-probability
            pred_marker = pred.eq(target.view_as(pred))   
            wrong_idx = (pred_marker == False).nonzero()  # get indices for wrong predictions
            for idx in wrong_idx:
              index = idx[0].item()
              title = "True:{},\n Pred:{}".format(classes[target[index].item()], classes[pred[index][0].item()])
              #print(title)
              ax = fig.add_subplot(5, 5, misclassifiedcounter+1, xticks=[], yticks=[])
              #ax.axis('off')
              ax.set_title(title)
              #plt.imshow(data[index].cpu().numpy().squeeze(), cmap='gray_r')
              self.imshow(data[index].cpu())
              
              misclassifiedcounter += 1
              if(misclassifiedcounter==numofimages):
                break
            
            if(misclassifiedcounter==numofimages):
                break
            
        fig.tight_layout()
        fig.savefig("{}.png".format(savefilename))
        return
    
    def savemisclassifiedimages(self,model, device, classes, testloader, outputdirectory, numofimages = 25):
        model.eval()
        os.makedirs(outputdirectory, exist_ok=True)
        
        misclassifiedimagenames = []
        misclassifiedtitles = []
        misclassifiedcounter = 0
        fig = plt.figure(figsize=(10,9))
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)     # get the index of the max log-probability
            pred_marker = pred.eq(target.view_as(pred))   
            wrong_idx = (pred_marker == False).nonzero()  # get indices for wrong predictions
            for idx in wrong_idx:
              index = idx[0].item()
              title = "True:{},\n Pred:{}".format(classes[target[index].item()], classes[pred[index][0].item()])
              misclassifiedtitles.append(title)
              # print(title)
              ax = fig.add_subplot(5, 5, misclassifiedcounter+1, xticks=[], yticks=[])
              # ax.axis('off')
              ax.set_title(title)
              # plt.imshow(data[index].cpu().numpy().squeeze(), cmap='gray_r')
              self.imshow(data[index].cpu())
    
              outputname = "{}.jpg".format(str(misclassifiedcounter))
              misclassifiedimagenames.append(outputname)
              outputpath = os.path.join(outputdirectory, outputname)
    
              save_image(data[index].cpu(), outputpath)
              
              misclassifiedcounter += 1
              if(misclassifiedcounter==numofimages):
                break
            
            if(misclassifiedcounter==numofimages):
                break
            
        fig.tight_layout()
        fig.savefig("misclassified.png")
        return (misclassifiedimagenames, misclassifiedtitles)
