import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as T

train = A.Compose([
    A.HorizontalFlip(p=1),
    T.ToTensor(),
    A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test = A.Compose([
    T.ToTensor(),
    A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class AlbumentationTransformTrain:
    def __init__(self):
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=1),
            A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            T.ToTensor()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img

class AlbumentationTransformTest:
    def __init__(self):

        self.transform = A.Compose([
            A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            T.ToTensor()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img