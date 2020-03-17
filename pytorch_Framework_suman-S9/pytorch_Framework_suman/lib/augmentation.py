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
import torch
import numpy as np


class AlbumentationTransformTrain:
    def __init__(self):
        #print(dir(A))
        self.transform = A.Compose([
            A.HorizontalFlip(p=1),
            #A.Rotate(limit=90),
            A.ShiftScaleRotate(),
            #A.RGBShift(),
            #A.RandomBrightnessContrast(),
            #A.GaussNoise(),
            A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=[0.4914, 0.4822, 0.4465], always_apply=False, p=0.5),
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

