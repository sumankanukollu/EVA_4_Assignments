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



from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
from zipfile import ZipFile

import torchvision

def show(tensors, figsize=(100,100), *args, **kwargs):
    grid_tensor = torchvision.utils.make_grid(tensors[:8], *args, **kwargs)
    grid_image = grid_tensor.permute(1,2,0)
    plt.figure(figsize=figsize)
    plt.imshow(grid_image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.close()


def saveplot(tensors, name, figsize=(100,100), *args, **kwargs):
	grid_tensor = torchvision.utils.make_grid(tensors[:8], *args, **kwargs)
	grid_image=grid_tensor.permute(1,2,0)
	plt.figure(figsize=figsize)
	plt.imshow((grid_image))
	plt.xticks([])
	plt.yticks([])
	plt.savefig(name, bbox_inches='tight')
	plt.close()