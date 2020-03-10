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