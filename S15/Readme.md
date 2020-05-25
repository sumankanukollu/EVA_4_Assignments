# Image "Depth-Map" and "Mask" prediction In single Model

 ![BGFG](https://github.com/sumankanukollu/EVA_4_Assignments/blob/master/S15/dataset_logs/5_test_bgfg.png)

![Mask Image](https://github.com/sumankanukollu/EVA_4_Assignments/blob/master/S15/dataset_logs/5_test_predicted_mask.png)

![Depthmap](https://github.com/sumankanukollu/EVA_4_Assignments/blob/master/S15/dataset_logs/5_test_predicted_depth.png)



# Objective:

Here I am trying to solve an open problem, i.e., for the given image (with **foreground object** and **background image**), 

* predict the depth map 
* as well as a mask, in a single model. 

This repository contains simple PyTorch implementations of **U-Net**, trained on [600K custom dataset](https://drive.google.com/drive/folders/11dkmzwk3CbB9utnlz-G2yLzvGb71E0zN?usp=sharing).

# Github Link: 

​		https://github.com/sumankanukollu/EVA_4_Assignments/tree/master/S15

## Approach to achieve:

Segregated the task into smaller steps as described below:

* [**Step-1:**](#Custom-Dataset-Preparation) Custom Dataset Preparation
    * [data statistcs and folder structure](#Data Statistics :bar_chart:)

* [**Step-2:**](#Strategy-to-work-with-Huge-Dataset:-*(in-Google-Drive)*)  Strategy to work with Huge Dataset: *(in Google Drive)*

* [**Step-3:**](#Next-Step-is-to-start-writing-Model:)  Next Step is to start writing Model

  * [Model Summary](#Model-Summary:)

* [**Step-4:**](#Train-and-Validation:)  Train and Validation

* [**Step-5**](#Code-Modularity:) Code Modularity

* [**Step-6:**](#Execution-commands:)  Execution commands

* [**Step-7:**](#Concepts-Explored:)  Concepts Explored


# Source code links:

* Github Link : https://github.com/sumankanukollu/EVA_4_Assignments/tree/master/S15
* [Script to create dataset](https://github.com/sumankanukollu/EVA_4_Assignments/blob/master/S15/scripts/pythonFiles/s15_datasetScript.ipynb)
* [Script to Train and Validation](https://github.com/sumankanukollu/EVA_4_Assignments/blob/master/S15/scripts/colabFiles/EVA4_S15_Final_Mask_Depth_v1.ipynb)

# **Custom Dataset Preparation**

* **Dataset Size** : **600K**  = (**200K** - BG-FG Images + **200K** Mask + **200K** Depth Images)
* Got to know that few modifications are needed with respect to the dataset, which I have created in Session-14, as part of Assignment-15A, so started with that and incorporated changes in scripts.
* **What I achieved?** : Time took to generate the dataset with new modifications **is reduced to 5 Hours** (from 4-days, in Assignment-15A) 
* Access the script from [here](https://github.com/sumankanukollu/EVA_4_Assignments/blob/master/S15/scripts/pythonFiles/s15_datasetScript.ipynb)

[back](https://github.com/sumankanukollu/EVA_4_Assignments/edit/master/S15/Readme.md#Approach-to-achieve:)

## Data Statistics :bar_chart:

* Whole <u>600K dataset</u> is divided into <u>20-Zip Files</u>, each one is having 30K images (30K * 20 = 600K).

* Each Zip file is having <u>30K Images</u> (10K+10K+10K) of (BGFG + Mask + Depth Map) and one **labels.txt** file.

* **Total Dataset size :**  <u>3.9 GB</u> 

* Each zip file is having around ~ 195 MB

* [link to Script](https://github.com/sumankanukollu/EVA_4_Assignments/blob/master/S15/scripts/pythonFiles/s15_datasetScript.ipynb)

* [600K dataset can be accessed from here](https://drive.google.com/drive/folders/11dkmzwk3CbB9utnlz-G2yLzvGb71E0zN?usp=sharing)

 [back](https://github.com/sumankanukollu/EVA_4_Assignments/edit/master/S15/Readme.md#Approach-to-achieve:)

  ### **:pushpin: Folder Structure**
  ```
      dataset
          ├── fg
          |   └───── 1_fg.jpg
          |   └───── 2_fg.jpg
          |   └───── .... 
          |   └───── 100_fg.jpg
          |
          |── bg
          |   └───── 1_bg.jpg
          |   └───── 2_bg. jpg
          |   └───── ....
          |   └───── 100_bg.jpg
          |
          |── mask
          |   └───── 1_fg_mask.jpg
          |   └───── 2_fg_mask.jpg
          |   └───── ....
          |   └───── 100_fg_mask.jpg
          |
          |── zipFiles
          |   └───── batch_1.zip
          |   |       |      └───── bg_fg_1
          |   |       |      |     └───── 1_bg_1_fg.jpg.jpg  to 5_bg_100_fg.jpg   (5*100*20 = 10K Images)
          |   |       |      └──── bg_fg_mask_1
          |   |       |      |     └───── 1_bg_1_fg.jpg.jpg  to 5_bg_100_fg.jpg   (5*100*20 = 10K Images)
          |   |       |      └──── depthMap
          |   |       |      |     └──────1_bg_1_fg.jpg.jpg  to 5_bg_100_fg.jpg   (5*100*20 = 10K Images)
          |   └───── batch_2.zip
          |   |       |      └───── bg_fg_1
          |   |       |      |     └───── 1_bg_1_fg.jpg.jpg  to 5_bg_100_fg.jpg   (5*100*20 = 10K Images)
          |   |       |      └──── bg_fg_mask_1
          |   |       |      |     └───── 1_bg_1_fg.jpg.jpg  to 5_bg_100_fg.jpg   (5*100*20 = 10K Images)
          |   |       |      └──── depthMap
          |   |       |      |     └────── 1_bg_1_fg.jpg.jpg  to 5_bg_100_fg.jpg   (5*100*20 = 10K Images)
          |   └────── .....
          |
          |   └───── batch_20.zip
          |   |       |      └───── bg_fg_1
          |   |       |      |     └───── 1_bg_1_fg.jpg.jpg  to 5_bg_100_fg.jpg   (5*100*20 = 10K Images)
          |   |       |      └──── bg_fg_mask_1
          |   |       |      |     └───── 1_bg_1_fg.jpg.jpg  to 5_bg_100_fg.jpg   (5*100*20 = 10K Images)
          |   |       |      └──── depthMap
          |   |       |      |     └────── 1_bg_1_fg.jpg.jpg  to 5_bg_100_fg.jpg   (5*100*20 = 10K Images)
          |
  ```

  ```
  Batch_1 zip file contains : 30001 files   >>> 30K Images + 1 labels.txt file 
  Batch_2 zip file contains : 30001 files
  Batch_3 zip file contains : 30001 files
  Batch_4 zip file contains : 30001 files
  Batch_5 zip file contains : 30001 files
  Batch_6 zip file contains : 30001 files
  Batch_7 zip file contains : 30001 files
  Batch_8 zip file contains : 30001 files
  Batch_9 zip file contains : 30001 files
  Batch_10 zip file contains : 30001 files
  Batch_11 zip file contains : 30001 files
  Batch_12 zip file contains : 30001 files
  Batch_13 zip file contains : 30001 files
  Batch_14 zip file contains : 30001 files
  Batch_15 zip file contains : 30001 files
  Batch_16 zip file contains : 30001 files
  Batch_17 zip file contains : 30001 files
  Batch_18 zip file contains : 30001 files
  Batch_19 zip file contains : 30001 files
  Batch_20 zip file contains : 30001 files
  Total Number of files present in dataset : 600020
  time: 40.7 s
  ```

[back](https://github.com/sumankanukollu/EVA_4_Assignments/edit/master/S15/Readme.md#Approach-to-achieve:)

# Strategy to work with Huge Dataset: *(in Google Drive)*

* Another major hurdle is working with huge dataset in GDrive and Colab environment.

* Spent more time on this, and explored **many ways with h5, binary files, and pickle files** to store this custom dataset.

* Observed that when we convert the dataset images into ND-Arrays , file size is more that what the actual size is.

  * <u>2K Images</u> are of size (224 * 224 * 3)   In folder = 20MB
    * **In binary format** (ND-Arrays)						  = 287 MB
    * **In tar.7Z format** (compression)                      = 28.3 MB
  * <u>200 Images</u> are of size (224 * 224 * 3)   In folder =1.17 MB
    * Which are **in pickle format** occupied of size   = 13 MB 

* So finally **decided to read Images directly from the generated zip files** (without extraction see code snippet below)

* So generated 20-Zip files as explained [above](##Data Statistics :bar_chart:) 

  ```python
  class customDataset(Dataset):
      def __init__(self, zipFName,transfrm):
          self.z_obj      = ZipFile(zipFName)
          self.transfrm   = transfrm
          # BG Images : dataPath.parent.joinpath('bg')
          self.bg         = os.listdir('/content/drive/My Drive/EVA4/S15/dataset/bg')
  
          # bgfg
          tmp             = list(filter(lambda x : x.startswith('bg_fg_1/'),self.z_obj.namelist()))
          tmp.remove('bg_fg_1/') if 'bg_fg_1/' in tmp else tmp
          self.bgfgF      = tmp
          del tmp
          # masks
          tmp             = list(filter(lambda x : x.startswith('bg_fg_mask_1/'),self.z_obj.namelist()))
          tmp.remove('bg_fg_mask_1/') if 'bg_fg_mask_1/' in tmp else tmp
          self.maskF      = tmp
          del tmp
          #depth
          tmp             =   list(filter(lambda x : x.startswith('depthMap/'),self.z_obj.namelist()))
          tmp.remove('depthMap/') if 'depthMap/' in tmp else tmp
          self.depthF     = tmp
          del tmp
  
      def __len__(self):
          return len(self.bgfgF)
  
      def __getitem__(self, idx):
          bgname = os.path.basename(self.bgfgF[idx]).split('_bg_')[0]+'_bg.jpg'
          bgF    = os.path.join('/content/drive/My Drive/EVA4/S15/dataset/bg',bgname)
  
          bgImg   = self.transfrm(Image.open(bgF))
          bgfgImg = self.transfrm(Image.open((self.z_obj.open(self.bgfgF[idx]))))
          maskImg = self.transfrm(Image.open((self.z_obj.open(self.maskF[idx]))))
          depthImg = self.transfrm(Image.open((self.z_obj.open(self.depthF[idx]))).convert('L'))
  
          return {'bgK' : bgImg,'bgfgK': bgfgImg, 'maskK' : maskImg, 'depthK': depthImg}
  
      def __del__(self):
          self.z_obj.close()
          del self.bgfgF
  
  ```

  [back](https://github.com/sumankanukollu/EVA_4_Assignments/edit/master/S15/Readme.md#Approach-to-achieve:)

# Next Step is to start writing Model:

* 1st question comes to my mind is which model I have to choose??

* Explored U-Net, Densenet and Resnet architectures.

* Implemented few basic models and started working with <u>small dataset of size 3000 Images</u>.

* As the **task was to identify masks**, we needed a network that can preserve edges and gradients predicted in the initial layers and use it in later layers for getting better information about the edges, <u>U-net was the closest architecture</u> to achieve same.

* **<u>My model brief</u>:**

  * Architecture is based on U-Net which is mainly used for mask detection/image segmentation
  * The architecture has layers for down sample and up sample.
  * Output from initial layers are added to layers during up sampling.
  * Initial layers are meant for detecting edges/gradients and this information is added to last layers for better prediction.
  * **Challenge** here  is to <u>*feed two inputs*</u> (bg and bg_fg) and *<u>get two outputs</u>* (mask and depth)
  * Input is 3 channel image, mask 1 channel and depth 1 channel.
  * To get two different outputs, tweaked final convolution layers as per required channels.
  * **Initially my model has 47 Million parameters**, after I **reduced the number of parameters to 12Million**.
  * Model did fairly well in mask but not so good for depth estimation, have to do some modifications.

  [back](https://github.com/sumankanukollu/EVA_4_Assignments/edit/master/S15/Readme.md#Approach-to-achieve:)

  # Model Summary:

  ```python
  ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  ================================================================
              Conv2d-1         [-1, 64, 112, 112]          18,816
         BatchNorm2d-2         [-1, 64, 112, 112]             128
                ReLU-3         [-1, 64, 112, 112]               0
           MaxPool2d-4           [-1, 64, 56, 56]               0
     ConvTranspose2d-5        [-1, 128, 111, 111]           8,192
              Conv2d-6          [-1, 128, 28, 28]          73,728
              Conv2d-7          [-1, 128, 28, 28]          16,384
         BatchNorm2d-8          [-1, 128, 28, 28]             256
                ReLU-9          [-1, 128, 28, 28]               0
             Conv2d-10          [-1, 128, 14, 14]         147,456
             Conv2d-11          [-1, 128, 14, 14]          16,384
        BatchNorm2d-12          [-1, 128, 14, 14]             256
               ReLU-13          [-1, 128, 14, 14]               0
           Downsize-14          [-1, 128, 14, 14]               0
    ConvTranspose2d-15          [-1, 256, 27, 27]          32,768
             Conv2d-16            [-1, 256, 7, 7]         294,912
             Conv2d-17            [-1, 256, 7, 7]          65,536
        BatchNorm2d-18            [-1, 256, 7, 7]             512
               ReLU-19            [-1, 256, 7, 7]               0
             Conv2d-20            [-1, 256, 4, 4]         589,824
             Conv2d-21            [-1, 256, 4, 4]          65,536
        BatchNorm2d-22            [-1, 256, 4, 4]             512
               ReLU-23            [-1, 256, 4, 4]               0
           Downsize-24            [-1, 256, 4, 4]               0
    ConvTranspose2d-25            [-1, 512, 7, 7]         131,072
             Conv2d-26            [-1, 512, 2, 2]       1,179,648
             Conv2d-27            [-1, 512, 2, 2]         262,144
        BatchNorm2d-28            [-1, 512, 2, 2]           1,024
               ReLU-29            [-1, 512, 2, 2]               0
             Conv2d-30            [-1, 512, 1, 1]       2,359,296
             Conv2d-31            [-1, 512, 1, 1]         262,144
        BatchNorm2d-32            [-1, 512, 1, 1]           1,024
               ReLU-33            [-1, 512, 1, 1]               0
           Downsize-34            [-1, 512, 1, 1]               0
    ConvTranspose2d-35            [-1, 256, 4, 4]       1,179,648
             Conv2d-36            [-1, 256, 4, 4]         589,824
             Conv2d-37            [-1, 256, 4, 4]          65,536
        BatchNorm2d-38            [-1, 256, 4, 4]             512
               ReLU-39            [-1, 256, 4, 4]               0
             Conv2d-40            [-1, 256, 4, 4]         589,824
             Conv2d-41            [-1, 256, 4, 4]          65,536
        BatchNorm2d-42            [-1, 256, 4, 4]             512
               ReLU-43            [-1, 256, 4, 4]               0
             Upsize-44            [-1, 256, 4, 4]               0
    ConvTranspose2d-45          [-1, 128, 28, 28]         294,912
             Conv2d-46          [-1, 128, 28, 28]         147,456
             Conv2d-47          [-1, 128, 28, 28]          16,384
        BatchNorm2d-48          [-1, 128, 28, 28]             256
               ReLU-49          [-1, 128, 28, 28]               0
             Conv2d-50          [-1, 128, 28, 28]         147,456
             Conv2d-51          [-1, 128, 28, 28]          16,384
        BatchNorm2d-52          [-1, 128, 28, 28]             256
               ReLU-53          [-1, 128, 28, 28]               0
             Upsize-54          [-1, 128, 28, 28]               0
    ConvTranspose2d-55           [-1, 64, 56, 56]          73,728
             Conv2d-56           [-1, 64, 56, 56]          36,864
             Conv2d-57           [-1, 64, 56, 56]           4,096
        BatchNorm2d-58           [-1, 64, 56, 56]             128
               ReLU-59           [-1, 64, 56, 56]               0
             Conv2d-60           [-1, 64, 56, 56]          36,864
             Conv2d-61           [-1, 64, 56, 56]           4,096
        BatchNorm2d-62           [-1, 64, 56, 56]             128
               ReLU-63           [-1, 64, 56, 56]               0
             Upsize-64           [-1, 64, 56, 56]               0
    ConvTranspose2d-65         [-1, 64, 112, 112]          36,864
             Conv2d-66         [-1, 64, 112, 112]          36,864
             Conv2d-67         [-1, 64, 112, 112]           4,096
        BatchNorm2d-68         [-1, 64, 112, 112]             128
               ReLU-69         [-1, 64, 112, 112]               0
             Conv2d-70         [-1, 64, 112, 112]          36,864
             Conv2d-71         [-1, 64, 112, 112]           4,096
        BatchNorm2d-72         [-1, 64, 112, 112]             128
               ReLU-73         [-1, 64, 112, 112]               0
             Upsize-74         [-1, 64, 112, 112]               0
    ConvTranspose2d-75         [-1, 32, 224, 224]          18,432
             Conv2d-76         [-1, 32, 224, 224]           9,216
             Conv2d-77         [-1, 32, 224, 224]           1,024
        BatchNorm2d-78         [-1, 32, 224, 224]              64
               ReLU-79         [-1, 32, 224, 224]               0
             Conv2d-80         [-1, 32, 224, 224]           9,216
             Conv2d-81         [-1, 32, 224, 224]           1,024
        BatchNorm2d-82         [-1, 32, 224, 224]              64
               ReLU-83         [-1, 32, 224, 224]               0
             Upsize-84         [-1, 32, 224, 224]               0
             Conv2d-85          [-1, 1, 224, 224]              32
    ConvTranspose2d-86            [-1, 256, 4, 4]       1,179,648
             Conv2d-87            [-1, 256, 4, 4]         589,824
             Conv2d-88            [-1, 256, 4, 4]          65,536
        BatchNorm2d-89            [-1, 256, 4, 4]             512
               ReLU-90            [-1, 256, 4, 4]               0
             Conv2d-91            [-1, 256, 4, 4]         589,824
             Conv2d-92            [-1, 256, 4, 4]          65,536
        BatchNorm2d-93            [-1, 256, 4, 4]             512
               ReLU-94            [-1, 256, 4, 4]               0
             Upsize-95            [-1, 256, 4, 4]               0
    ConvTranspose2d-96          [-1, 128, 28, 28]         294,912
             Conv2d-97          [-1, 128, 28, 28]         147,456
             Conv2d-98          [-1, 128, 28, 28]          16,384
        BatchNorm2d-99          [-1, 128, 28, 28]             256
              ReLU-100          [-1, 128, 28, 28]               0
            Conv2d-101          [-1, 128, 28, 28]         147,456
            Conv2d-102          [-1, 128, 28, 28]          16,384
       BatchNorm2d-103          [-1, 128, 28, 28]             256
              ReLU-104          [-1, 128, 28, 28]               0
            Upsize-105          [-1, 128, 28, 28]               0
   ConvTranspose2d-106           [-1, 64, 56, 56]          73,728
            Conv2d-107           [-1, 64, 56, 56]          36,864
            Conv2d-108           [-1, 64, 56, 56]           4,096
       BatchNorm2d-109           [-1, 64, 56, 56]             128
              ReLU-110           [-1, 64, 56, 56]               0
            Conv2d-111           [-1, 64, 56, 56]          36,864
            Conv2d-112           [-1, 64, 56, 56]           4,096
       BatchNorm2d-113           [-1, 64, 56, 56]             128
              ReLU-114           [-1, 64, 56, 56]               0
            Upsize-115           [-1, 64, 56, 56]               0
   ConvTranspose2d-116         [-1, 64, 112, 112]          36,864
            Conv2d-117         [-1, 64, 112, 112]          36,864
            Conv2d-118         [-1, 64, 112, 112]           4,096
       BatchNorm2d-119         [-1, 64, 112, 112]             128
              ReLU-120         [-1, 64, 112, 112]               0
            Conv2d-121         [-1, 64, 112, 112]          36,864
            Conv2d-122         [-1, 64, 112, 112]           4,096
       BatchNorm2d-123         [-1, 64, 112, 112]             128
              ReLU-124         [-1, 64, 112, 112]               0
            Upsize-125         [-1, 64, 112, 112]               0
   ConvTranspose2d-126         [-1, 32, 224, 224]          18,432
            Conv2d-127         [-1, 32, 224, 224]           9,216
            Conv2d-128         [-1, 32, 224, 224]           1,024
       BatchNorm2d-129         [-1, 32, 224, 224]              64
              ReLU-130         [-1, 32, 224, 224]               0
            Conv2d-131         [-1, 32, 224, 224]           9,216
            Conv2d-132         [-1, 32, 224, 224]           1,024
       BatchNorm2d-133         [-1, 32, 224, 224]              64
              ReLU-134         [-1, 32, 224, 224]               0
            Upsize-135         [-1, 32, 224, 224]               0
            Conv2d-136          [-1, 3, 224, 224]              99
  ================================================================
  Total params: 12,384,643
  Trainable params: 12,384,643
  Non-trainable params: 0
  ----------------------------------------------------------------
  Input size (MB): 1.15
  Forward/backward pass size (MB): 453.79
  Params size (MB): 47.24
  Estimated Total Size (MB): 502.18
  ----------------------------------------------------------------
  ```

[back](https://github.com/sumankanukollu/EVA_4_Assignments/edit/master/S15/Readme.md#Approach-to-achieve:)

# Train and Validation: 

* Training and Validation has been done on each zipfile separately.
* Full data set was split into 70% and 30% for train and test respectively using the torch random_split technique.
* **Batch Size** : 50 
* Observed that 1st epoch took more time ~ 33 minutes, after that epoch time is reduced.
* **Time took To complete 5-Epochs :** 20 minutes 
* **Strategy :** 
  * Initial loss value is defined as : 1e+5
  * After 1st Train epoch saving the model weights using "torch.save" , with the new loss value as "Val_loss_min".
  * Validating the results
  * After 2nd Train epoch again I am validating the 2nd epoch "loss value" with "Val_loss_min" value.
    * If "loss value" < "Val_loss_min", then update the "Val_loss_min" value with the "loss" value.
    * Save the model state with new "Val_loss_min" value.
  * Repeat the loop again for 3rd, 4th ......
  * Each time if loss < Val_loss_min, save the checkpoint as best model.
  * Load this checkpoint with "torch.load" for next run. (Can see the logic in below train and validation logs)

## Summary of Training cycles:

| S.No    | Zip File Name | Train Sz | Test Sz | Batch Sz | Epochs          |
| ------- | ------------- | -------- | ------- | -------- | --------------- |
| 1       | batch_1.zip   | 21K      | 7K      | 50       | 5               |
| 2       | batch_2.zip   | 21K      | 7K      | 50       | 5               |
| 3       | batch_3.zip   | 21K      | 7K      | 50       | 5               |
| ....... | .......       | .......  | ....... | .......  | .......         |
| ....... | .......       | .......  | ....... | .......  | .......         |
| 20      | batch_20.zip  | 21K      | 7K      | 50       | 5               |
|         |               |          |         |          | **Total : 100** |

## Train and Validation Logs:

```
You are executing model on : cuda
Dataset using : /content/drive/My Drive/EVA4/S15/dataset/zipFiles/batch_11.zip
Train Dataset : 21000
Test Dataset : 9000
	 In batch size of : 50
cuda

You are running /content/drive/My Drive/EVA4/S15/dataset/zipFiles/batch_11.zip dataset for 10 epochs

###########EPOCH:1#########

  0% 0/140 [00:00<?, ?it/s]

######################
You are in Train process
################################
Overall Loss=2.2377 Mask loss=0.71 Depth loss=0.81:   0% 0/140 [00:48<?, ?it/s]batch_idx: 0
loss: 2.237684726715088
##### Train Epoch: 1  [0/7000  (0%)]	Loss:2.237685
Overall Loss=0.4396 Mask loss=0.04 Depth loss=0.37:  36% 50/140 [08:43<03:09,  2.10s/it]batch_idx: 50
loss: 0.4395734369754791
##### Train Epoch: 1  [200/7000  (36%)]	Loss:0.439573
Overall Loss=0.4045 Mask loss=0.02 Depth loss=0.35:  71% 100/140 [09:34<00:34,  1.16it/s]batch_idx: 100
loss: 0.40453237295150757
##### Train Epoch: 1  [400/7000  (71%)]	Loss:0.404532
Overall Loss=0.3997 Mask loss=0.02 Depth loss=0.35: 100% 140/140 [10:01<00:00,  4.29s/it]
After compleation of training at epoch :139
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Validation loss decreased (10000.000000 --> 0.399683).  Saving model ...



**************Model saved**************





######################
You are in Test process
################################
  0% 0/60 [00:00<?, ?it/s]batch_idx: 0
loss: 0.4114297106862068
 83% 50/60 [00:31<00:06,  1.55it/s]batch_idx: 50
loss: 0.41259292140603065
100% 60/60 [00:37<00:00,  1.60it/s]

####################
Test set: Avg loss: 0.4121, Mask Loss: 0.02, Depth Loss: 0.37
############
After compleation of Test at epoch :59
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

###########EPOCH:2#########

  0% 0/140 [00:00<?, ?it/s]

######################
You are in Train process
################################
Overall Loss=0.4001 Mask loss=0.02 Depth loss=0.36:   0% 0/140 [00:00<?, ?it/s]batch_idx: 0
loss: 0.40005800127983093
##### Train Epoch: 2  [0/7000  (0%)]	Loss:0.400058
Overall Loss=0.3892 Mask loss=0.02 Depth loss=0.35:  36% 50/140 [00:33<00:57,  1.56it/s]batch_idx: 50
loss: 0.3891509771347046
##### Train Epoch: 2  [200/7000  (36%)]	Loss:0.389151
Overall Loss=0.3780 Mask loss=0.02 Depth loss=0.34:  71% 100/140 [01:05<00:25,  1.58it/s]batch_idx: 100
loss: 0.37801164388656616
##### Train Epoch: 2  [400/7000  (71%)]	Loss:0.378012
Overall Loss=0.3822 Mask loss=0.02 Depth loss=0.35: 100% 140/140 [01:30<00:00,  1.54it/s]
After compleation of training at epoch :139
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Validation loss decreased (0.399683 --> 0.382165).  Saving model ...



**************Model saved**************





######################
You are in Test process
################################
  0% 0/60 [00:00<?, ?it/s]batch_idx: 0
loss: 0.39386575296521187
 83% 50/60 [00:30<00:05,  1.67it/s]batch_idx: 50
loss: 0.390657652169466
100% 60/60 [00:36<00:00,  1.66it/s]

####################
Test set: Avg loss: 0.3935, Mask Loss: 0.02, Depth Loss: 0.36
############
After compleation of Test at epoch :59
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

###########EPOCH:3#########

  0% 0/140 [00:00<?, ?it/s]

######################
You are in Train process
################################
Overall Loss=0.3821 Mask loss=0.02 Depth loss=0.35:   0% 0/140 [00:00<?, ?it/s]batch_idx: 0
loss: 0.3821311891078949
##### Train Epoch: 3  [0/7000  (0%)]	Loss:0.382131
Overall Loss=0.3815 Mask loss=0.02 Depth loss=0.35:  36% 50/140 [00:33<00:59,  1.52it/s]batch_idx: 50
loss: 0.3814584016799927
##### Train Epoch: 3  [200/7000  (36%)]	Loss:0.381458
Overall Loss=0.3820 Mask loss=0.02 Depth loss=0.35:  71% 100/140 [01:05<00:25,  1.58it/s]batch_idx: 100
loss: 0.3820171356201172
##### Train Epoch: 3  [400/7000  (71%)]	Loss:0.382017
Overall Loss=0.3751 Mask loss=0.02 Depth loss=0.34: 100% 140/140 [01:30<00:00,  1.54it/s]
After compleation of training at epoch :139
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Validation loss decreased (0.382165 --> 0.375150).  Saving model ...



**************Model saved**************





######################
You are in Test process
################################
  0% 0/60 [00:00<?, ?it/s]batch_idx: 0
loss: 0.41060758382081985
 83% 50/60 [00:30<00:06,  1.65it/s]batch_idx: 50
loss: 0.4116758480668068
100% 60/60 [00:36<00:00,  1.66it/s]

####################
Test set: Avg loss: 0.4142, Mask Loss: 0.03, Depth Loss: 0.36
############
After compleation of Test at epoch :59
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

###########EPOCH:4#########

  0% 0/140 [00:00<?, ?it/s]

######################
You are in Train process
################################
Overall Loss=0.3743 Mask loss=0.01 Depth loss=0.35:   0% 0/140 [00:00<?, ?it/s]batch_idx: 0
loss: 0.3742503523826599
##### Train Epoch: 4  [0/7000  (0%)]	Loss:0.374250
Overall Loss=0.3724 Mask loss=0.01 Depth loss=0.34:  36% 50/140 [00:32<00:57,  1.57it/s]batch_idx: 50
loss: 0.3723639249801636
##### Train Epoch: 4  [200/7000  (36%)]	Loss:0.372364
Overall Loss=0.3753 Mask loss=0.01 Depth loss=0.35:  71% 100/140 [01:04<00:25,  1.56it/s]batch_idx: 100
loss: 0.3753097951412201
##### Train Epoch: 4  [400/7000  (71%)]	Loss:0.375310
Overall Loss=0.3756 Mask loss=0.01 Depth loss=0.35: 100% 140/140 [01:30<00:00,  1.55it/s]
After compleation of training at epoch :139
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


######################
You are in Test process
################################
  0% 0/60 [00:00<?, ?it/s]batch_idx: 0
loss: 0.5483980625867844
 83% 50/60 [00:30<00:05,  1.69it/s]batch_idx: 50
loss: 0.5647959858179092
100% 60/60 [00:36<00:00,  1.66it/s]

####################
Test set: Avg loss: 0.5619, Mask Loss: 0.10, Depth Loss: 0.36
############
After compleation of Test at epoch :59
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

###########EPOCH:5#########

  0% 0/140 [00:00<?, ?it/s]

######################
You are in Train process
################################
Overall Loss=0.3779 Mask loss=0.02 Depth loss=0.35:   0% 0/140 [00:00<?, ?it/s]batch_idx: 0
loss: 0.37786269187927246
##### Train Epoch: 5  [0/7000  (0%)]	Loss:0.377863
Overall Loss=0.3759 Mask loss=0.01 Depth loss=0.35:  36% 50/140 [00:32<00:56,  1.58it/s]batch_idx: 50
loss: 0.37591761350631714
##### Train Epoch: 5  [200/7000  (36%)]	Loss:0.375918
Overall Loss=0.3758 Mask loss=0.01 Depth loss=0.35:  71% 100/140 [01:05<00:25,  1.58it/s]batch_idx: 100
loss: 0.3757675290107727
##### Train Epoch: 5  [400/7000  (71%)]	Loss:0.375768
Overall Loss=0.3768 Mask loss=0.01 Depth loss=0.35: 100% 140/140 [01:31<00:00,  1.54it/s]
After compleation of training at epoch :139
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


######################
You are in Test process
################################
  0% 0/60 [00:00<?, ?it/s]batch_idx: 0
loss: 0.394613865762949
 83% 50/60 [00:30<00:05,  1.68it/s]batch_idx: 50
loss: 0.39698198810219765
100% 60/60 [00:36<00:00,  1.66it/s]

####################
Test set: Avg loss: 0.3933, Mask Loss: 0.02, Depth Loss: 0.36
############
After compleation of Test at epoch :59
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

###########EPOCH:6#########

  0% 0/140 [00:00<?, ?it/s]

######################
You are in Train process
################################
Overall Loss=0.3689 Mask loss=0.01 Depth loss=0.34:   0% 0/140 [00:00<?, ?it/s]batch_idx: 0
loss: 0.36885249614715576
##### Train Epoch: 6  [0/7000  (0%)]	Loss:0.368852
Overall Loss=0.3712 Mask loss=0.01 Depth loss=0.34:  36% 50/140 [00:32<00:57,  1.58it/s]batch_idx: 50
loss: 0.37118980288505554
##### Train Epoch: 6  [200/7000  (36%)]	Loss:0.371190
Overall Loss=0.3727 Mask loss=0.01 Depth loss=0.34:  71% 100/140 [01:05<00:25,  1.58it/s]batch_idx: 100
loss: 0.37270042300224304
##### Train Epoch: 6  [400/7000  (71%)]	Loss:0.372700
Overall Loss=0.3667 Mask loss=0.01 Depth loss=0.34: 100% 140/140 [01:30<00:00,  1.55it/s]
After compleation of training at epoch :139
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Validation loss decreased (0.375150 --> 0.366716).  Saving model ...



**************Model saved**************





######################
You are in Test process
################################
  0% 0/60 [00:00<?, ?it/s]batch_idx: 0
loss: 0.3892221376299858
 83% 50/60 [00:30<00:05,  1.68it/s]batch_idx: 50
loss: 0.3924296498298645
100% 60/60 [00:36<00:00,  1.65it/s]

####################
Test set: Avg loss: 0.3920, Mask Loss: 0.02, Depth Loss: 0.36
############
After compleation of Test at epoch :59
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

###########EPOCH:7#########

  0% 0/140 [00:00<?, ?it/s]

######################
You are in Train process
################################
Overall Loss=0.3706 Mask loss=0.01 Depth loss=0.34:   0% 0/140 [00:00<?, ?it/s]batch_idx: 0
loss: 0.3705996870994568
##### Train Epoch: 7  [0/7000  (0%)]	Loss:0.370600
Overall Loss=0.3692 Mask loss=0.01 Depth loss=0.34:  36% 50/140 [00:33<00:58,  1.55it/s]batch_idx: 50
loss: 0.3691951036453247
##### Train Epoch: 7  [200/7000  (36%)]	Loss:0.369195
Overall Loss=0.3744 Mask loss=0.01 Depth loss=0.35:  71% 100/140 [01:05<00:26,  1.51it/s]batch_idx: 100
loss: 0.37437161803245544
##### Train Epoch: 7  [400/7000  (71%)]	Loss:0.374372
Overall Loss=0.3705 Mask loss=0.01 Depth loss=0.34: 100% 140/140 [01:31<00:00,  1.54it/s]
After compleation of training at epoch :139
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


######################
You are in Test process
################################
  0% 0/60 [00:00<?, ?it/s]batch_idx: 0
loss: 0.3938722126185894
 83% 50/60 [00:30<00:06,  1.63it/s]batch_idx: 50
loss: 0.3938399776816368
100% 60/60 [00:36<00:00,  1.66it/s]

####################
Test set: Avg loss: 0.3939, Mask Loss: 0.02, Depth Loss: 0.35
############
After compleation of Test at epoch :59
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

###########EPOCH:8#########

  0% 0/140 [00:00<?, ?it/s]

######################
You are in Train process
################################
Overall Loss=0.3700 Mask loss=0.01 Depth loss=0.34:   0% 0/140 [00:00<?, ?it/s]batch_idx: 0
loss: 0.37003403902053833
##### Train Epoch: 8  [0/7000  (0%)]	Loss:0.370034
Overall Loss=0.3659 Mask loss=0.01 Depth loss=0.34:  36% 50/140 [00:33<00:57,  1.58it/s]batch_idx: 50
loss: 0.36586135625839233
##### Train Epoch: 8  [200/7000  (36%)]	Loss:0.365861
Overall Loss=0.3706 Mask loss=0.01 Depth loss=0.35:  71% 100/140 [01:05<00:25,  1.57it/s]batch_idx: 100
loss: 0.3705718219280243
##### Train Epoch: 8  [400/7000  (71%)]	Loss:0.370572
Overall Loss=0.3640 Mask loss=0.01 Depth loss=0.34: 100% 140/140 [01:30<00:00,  1.54it/s]
After compleation of training at epoch :139
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Validation loss decreased (0.366716 --> 0.363993).  Saving model ...



**************Model saved**************





######################
You are in Test process
################################
  0% 0/60 [00:00<?, ?it/s]batch_idx: 0
loss: 0.39405376836657524
 83% 50/60 [00:30<00:05,  1.68it/s]batch_idx: 50
loss: 0.39544009417295456
100% 60/60 [00:36<00:00,  1.65it/s]

####################
Test set: Avg loss: 0.3936, Mask Loss: 0.02, Depth Loss: 0.36
############
After compleation of Test at epoch :59
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

###########EPOCH:9#########

  0% 0/140 [00:00<?, ?it/s]

######################
You are in Train process
################################
Overall Loss=0.3698 Mask loss=0.01 Depth loss=0.34:   0% 0/140 [00:00<?, ?it/s]batch_idx: 0
loss: 0.3698427081108093
##### Train Epoch: 9  [0/7000  (0%)]	Loss:0.369843
Overall Loss=0.3718 Mask loss=0.01 Depth loss=0.35:  36% 50/140 [00:33<00:57,  1.57it/s]batch_idx: 50
loss: 0.37184837460517883
##### Train Epoch: 9  [200/7000  (36%)]	Loss:0.371848
Overall Loss=0.3745 Mask loss=0.01 Depth loss=0.35:  71% 100/140 [01:05<00:25,  1.56it/s]batch_idx: 100
loss: 0.37454167008399963
##### Train Epoch: 9  [400/7000  (71%)]	Loss:0.374542
Overall Loss=0.3694 Mask loss=0.01 Depth loss=0.34: 100% 140/140 [01:31<00:00,  1.54it/s]
After compleation of training at epoch :139
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


######################
You are in Test process
################################
  0% 0/60 [00:00<?, ?it/s]batch_idx: 0
loss: 0.38471102342009544
 83% 50/60 [00:30<00:06,  1.65it/s]batch_idx: 50
loss: 0.38719100318849087
100% 60/60 [00:36<00:00,  1.66it/s]

####################
Test set: Avg loss: 0.3878, Mask Loss: 0.02, Depth Loss: 0.36
############
After compleation of Test at epoch :59
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

###########EPOCH:10#########

  0% 0/140 [00:00<?, ?it/s]

######################
You are in Train process
################################
Overall Loss=0.3724 Mask loss=0.01 Depth loss=0.35:   0% 0/140 [00:00<?, ?it/s]batch_idx: 0
loss: 0.3723539113998413
##### Train Epoch: 10  [0/7000  (0%)]	Loss:0.372354
Overall Loss=0.3686 Mask loss=0.01 Depth loss=0.34:  36% 50/140 [00:33<00:57,  1.56it/s]batch_idx: 50
loss: 0.3685966730117798
##### Train Epoch: 10  [200/7000  (36%)]	Loss:0.368597
Overall Loss=0.3731 Mask loss=0.01 Depth loss=0.35:  71% 100/140 [01:05<00:25,  1.58it/s]batch_idx: 100
loss: 0.373074471950531
##### Train Epoch: 10  [400/7000  (71%)]	Loss:0.373074
Overall Loss=0.3692 Mask loss=0.01 Depth loss=0.35: 100% 140/140 [01:30<00:00,  1.54it/s]
After compleation of training at epoch :139
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


######################
You are in Test process
################################
  0% 0/60 [00:00<?, ?it/s]batch_idx: 0
loss: 0.38916001841425896
 83% 50/60 [00:30<00:06,  1.66it/s]batch_idx: 50
loss: 0.38923441246151924
100% 60/60 [00:36<00:00,  1.65it/s]

####################
Test set: Avg loss: 0.3878, Mask Loss: 0.02, Depth Loss: 0.36
############
After compleation of Test at epoch :59
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
time: 33min 9s
```

[back](https://github.com/sumankanukollu/EVA_4_Assignments/edit/master/S15/Readme.md#Approach-to-achieve:)

## loss Functions used:

* BCE loss with logits (segmoid handled internally) is used for both image and text labels ground truth. 
* For image pixel based tasks, SSIM and even Dice loss can be considered. 
* Tried all 3;
  * (BCE with logits)
  * SSIM (with kernel size 3 and reduction method as mean) and 
  * Dice loss
*  BCE loss seems to be giving better result. With other 2 loss functions got black screen (may be something I missed)

# Code Modularity:

* Using the previous assignment code:

![Snippet](https://github.com/sumankanukollu/EVA_4_Assignments/blob/master/S15/dataset_logs/codeModularity.jpg)

```python
* Lib 			: contains all reusable modules
* ModelWeights	: used to save checkpoints and all loss values will be captured in pickle file 
* utils 		: contains all utility related methods
* Scripts 		: All scripts are placed in this folder
```



[back](https://github.com/sumankanukollu/EVA_4_Assignments/edit/master/S15/Readme.md#Approach-to-achieve:)



# Execution commands:

* ```
  os.chdir(r'/content/drive/My Drive/EVA4/S15/scripts/pythonFiles')
  !python train_mask_depth.py -z '/content/drive/My Drive/EVA4/S15/dataset/zipFiles/batch_1.zip' -e 5  -bs 50
  ```

  * -e    : Number of epochs 

  * -bs  : Batch Size

  * -z    : Input Zip file (which dataset we are using for train and validation) 

    

[back](https://github.com/sumankanukollu/EVA_4_Assignments/edit/master/S15/Readme.md#Approach-to-achieve:)



# Concepts Explored:

* Explored how CIFAR-10 dataset is prepared and loaded with 60K dataset, started from there 
* H5 File generation and load from the custom dataset
* Binary file generation and load  for the custom dataset in ND-Arrays
* Explored how Pickle is useful
* Writing Custom dataset in Pytorch 
  * Map-Style Dataset
  * Iterable Dataset
* Data Loader Usage
* Played with PIL, numpy modules and Torch Tensors
* Image conversions (jpg : to png, to binaty, to numpy arrays, and torch tensors, channel suppression, gray scale, )
  With these concepts now I can play with any kind of Image, got such confidence.
* How to handle CUDA-Out Of Memory Issues
* Dataset Compression techniques with ZipFile, tar, modules
* U-Net, DenseNet and ResNet architectures.
* Segmentation Techniques
* Loss Functions

[back](https://github.com/sumankanukollu/EVA_4_Assignments/edit/master/S15/Readme.md#Approach-to-achieve:)
