# EVA4 Assignment 11 - Suman Kanukollu

## Code-1: With pytorch :

- [Github link](https://github.com/sumankanukollu/EVA_4_Assignments/blob/S_10_LRFinder_Misclassified_Cifar10/EVA4_S11_Suman_pytorch_OCP.ipynb) 

- [Colab link](https://colab.research.google.com/drive/12MvzlskNOUP-tYEbFBRIMNjThPlciL_o?authuser=1#scrollTo=jtAc0XHsQ6gW)

## Code-2: With pytorch : With fast.ai 

- [Github link](https://github.com/sumankanukollu/EVA_4_Assignments/blob/S_10_LRFinder_Misclassified_Cifar10/EVA4_S11_fastAi_OCP_Suman.ipynb) 

- [Colab link](https://colab.research.google.com/drive/1XngrgLNtz19jrrrEVoZoMepuw_0MJGYs?authuser=1#scrollTo=E9I-0rsLofAx)


# Model Summary:


----------
- Model Name: **Session-11 Model in Resnet Arch**

- No of parameters: **6,573,130**

- No of Epochs ran: **24**

	## Accuracy:
	- **Max. Train Accuracy** : 95.548 at 24-Epoach 
	- **Max. Test Accuracy**  : 90.37 at 23-Epoach 
	## Loss:
	- **Min. Train Loss**     : 0.000260 at 24-Epoach 
	- **Min. Test Loss**      : 0.000598 at 24-Epoach 

	## Target (90%) Achieved at :
	- 24th epoch  : *Test accuracy is : 90.37*


# Framework Structure:
	  - EVA4_S11_Suman_pytorch_OCP.ipynb : It contains main workflow
	  - lib     : It contains all the functions related to Augmentation,Loaddataset,lrFinder,trainTestMethods
	  - models  : It contains S11 model
	  - utils   : It contains helper functions such as progress bar and random images display and saveimages
	  - Sampleimages : Where user can place images for testing the code
	  - Output       : Network results will be saved here  

# LR Range Test graph:
![LR Range Test Graph](https://github.com/sumankanukollu/EVA_4_Assignments/blob/S_10_LRFinder_Misclassified_Cifar10/pytorch_Framework_suman-S9/pytorch_Framework_suman/outputs/OCP_lr_rangeTestGraph.JPG)


# Training and Test Accuracy and Loss curves:
![image](https://github.com/sumankanukollu/EVA_4_Assignments/blob/S_10_LRFinder_Misclassified_Cifar10/pytorch_Framework_suman-S9/pytorch_Framework_suman/outputs/OCP_trainTest_Acc_lossGraph.JPG)

# OCP Triangle plot:
![image](https://github.com/sumankanukollu/EVA_4_Assignments/blob/S_10_LRFinder_Misclassified_Cifar10/pytorch_Framework_suman-S9/pytorch_Framework_suman/outputs/OCP_trianglePlot.jpg)

# OCP LR Plot:
![image](https://github.com/sumankanukollu/EVA_4_Assignments/blob/S_10_LRFinder_Misclassified_Cifar10/pytorch_Framework_suman-S9/pytorch_Framework_suman/outputs/OCP_lrPlot.JPG)

# Logs

  - **Best accuracy**
		  
	```
	
		EPOCH: 24 LR: 0.00020598437499999914

		Epoch: 24
		 [================================================================>]  Step: 160ms | Tot: 24s703ms | Train >> Loss: 0.132 | Acc: 95.548% (47774/50000) 98/98 
		 [=============================================================>...]  Step: 35ms | Tot: 1s370ms | Test >> Loss: 0.299 | Acc: 90.350% (9035/10000) 20/20 
		0.0005979749128222465



  - **Full Log**

    ```
    
			
		EPOCH: 1 LR: 0.001647875

		Epoch: 1
		 [================================================================>]  Step: 158ms | Tot: 25s248ms | Train >> Loss: 1.753 | Acc: 38.060% (19030/50000) 98/98 
		 [=============================================================>...]  Step: 35ms | Tot: 1s381ms | Test >> Loss: 1.261 | Acc: 54.090% (5409/10000) 20/20 
		0.0025219256639480592

		EPOCH: 2 LR: 0.004683434210526315

		Epoch: 2
		 [================================================================>]  Step: 158ms | Tot: 25s295ms | Train >> Loss: 1.238 | Acc: 55.878% (27939/50000) 98/98 
		 [=============================================================>...]  Step: 37ms | Tot: 1s425ms | Test >> Loss: 1.093 | Acc: 62.180% (6218/10000) 20/20 
		0.0021868449211120606

		EPOCH: 3 LR: 0.00771899342105263

		Epoch: 3
		 [================================================================>]  Step: 159ms | Tot: 25s323ms | Train >> Loss: 1.019 | Acc: 64.196% (32098/50000) 98/98 
		 [=============================================================>...]  Step: 35ms | Tot: 1s467ms | Test >> Loss: 0.861 | Acc: 70.240% (7024/10000) 20/20 
		0.001722007131576538

		EPOCH: 4 LR: 0.010754552631578946

		Epoch: 4
		 [================================================================>]  Step: 158ms | Tot: 25s108ms | Train >> Loss: 0.846 | Acc: 70.754% (35377/50000) 98/98 
		 [=============================================================>...]  Step: 33ms | Tot: 1s306ms | Test >> Loss: 0.884 | Acc: 71.450% (7145/10000) 20/20 
		0.0017675743162631988

		EPOCH: 5 LR: 0.01304782275390625

		Epoch: 5
		 [================================================================>]  Step: 159ms | Tot: 25s239ms | Train >> Loss: 0.754 | Acc: 73.910% (36955/50000) 98/98 
		 [=============================================================>...]  Step: 35ms | Tot: 1s487ms | Test >> Loss: 0.870 | Acc: 71.370% (7137/10000) 20/20 
		0.0017390096306800842

		EPOCH: 6 LR: 0.0123719365234375

		Epoch: 6
		 [================================================================>]  Step: 157ms | Tot: 25s90ms | Train >> Loss: 0.639 | Acc: 78.024% (39012/50000) 98/98 
		 [=============================================================>...]  Step: 39ms | Tot: 1s454ms | Test >> Loss: 0.633 | Acc: 78.680% (7868/10000) 20/20 
		0.0012650435090065003

		EPOCH: 7 LR: 0.011696050292968751

		Epoch: 7
		 [================================================================>]  Step: 157ms | Tot: 25s67ms | Train >> Loss: 0.570 | Acc: 80.314% (40157/50000) 98/98 
		 [=============================================================>...]  Step: 35ms | Tot: 1s389ms | Test >> Loss: 0.649 | Acc: 79.560% (7956/10000) 20/20 
		0.001298457145690918

		EPOCH: 8 LR: 0.0110201640625

		Epoch: 8
		 [================================================================>]  Step: 158ms | Tot: 25s158ms | Train >> Loss: 0.530 | Acc: 81.618% (40809/50000) 98/98 
		 [=============================================================>...]  Step: 35ms | Tot: 1s349ms | Test >> Loss: 0.488 | Acc: 83.170% (8317/10000) 20/20 
		0.0009755059450864792

		EPOCH: 9 LR: 0.01034427783203125

		Epoch: 9
		 [================================================================>]  Step: 159ms | Tot: 25s272ms | Train >> Loss: 0.468 | Acc: 83.730% (41865/50000) 98/98 
		 [=============================================================>...]  Step: 35ms | Tot: 1s415ms | Test >> Loss: 0.597 | Acc: 81.280% (8128/10000) 20/20 
		0.001194022673368454

		EPOCH: 10 LR: 0.0096683916015625

		Epoch: 10
		 [================================================================>]  Step: 159ms | Tot: 25s160ms | Train >> Loss: 0.437 | Acc: 84.732% (42366/50000) 98/98 
		 [=============================================================>...]  Step: 34ms | Tot: 1s545ms | Test >> Loss: 0.525 | Acc: 82.850% (8285/10000) 20/20 
		0.0010505159944295883

		EPOCH: 11 LR: 0.00899250537109375

		Epoch: 11
		 [================================================================>]  Step: 159ms | Tot: 24s834ms | Train >> Loss: 0.393 | Acc: 86.338% (43169/50000) 98/98 
		 [=============================================================>...]  Step: 36ms | Tot: 1s354ms | Test >> Loss: 0.468 | Acc: 84.160% (8416/10000) 20/20 
		0.0009352662771940231

		EPOCH: 12 LR: 0.008316619140625

		Epoch: 12
		 [================================================================>]  Step: 159ms | Tot: 25s85ms | Train >> Loss: 0.368 | Acc: 87.102% (43551/50000) 98/98 
		 [=============================================================>...]  Step: 35ms | Tot: 1s304ms | Test >> Loss: 0.424 | Acc: 86.370% (8637/10000) 20/20 
		0.0008477372229099274

		EPOCH: 13 LR: 0.00764073291015625

		Epoch: 13
		 [================================================================>]  Step: 158ms | Tot: 24s829ms | Train >> Loss: 0.344 | Acc: 87.876% (43938/50000) 98/98 
		 [=============================================================>...]  Step: 35ms | Tot: 1s364ms | Test >> Loss: 0.426 | Acc: 86.280% (8628/10000) 20/20 
		0.0008515920579433441

		EPOCH: 14 LR: 0.0069648466796875

		Epoch: 14
		 [================================================================>]  Step: 158ms | Tot: 24s937ms | Train >> Loss: 0.321 | Acc: 88.802% (44401/50000) 98/98 
		 [=============================================================>...]  Step: 34ms | Tot: 1s451ms | Test >> Loss: 0.433 | Acc: 85.930% (8593/10000) 20/20 
		0.0008669940263032913

		EPOCH: 15 LR: 0.0062889604492187496

		Epoch: 15
		 [================================================================>]  Step: 158ms | Tot: 25s135ms | Train >> Loss: 0.285 | Acc: 90.134% (45067/50000) 98/98 
		 [=============================================================>...]  Step: 34ms | Tot: 1s543ms | Test >> Loss: 0.362 | Acc: 87.840% (8784/10000) 20/20 
		0.0007242325633764267

		EPOCH: 16 LR: 0.005613074218749999

		Epoch: 16
		 [================================================================>]  Step: 158ms | Tot: 24s622ms | Train >> Loss: 0.270 | Acc: 90.608% (45304/50000) 98/98 
		 [=============================================================>...]  Step: 36ms | Tot: 1s367ms | Test >> Loss: 0.379 | Acc: 87.970% (8797/10000) 20/20 
		0.000757380548119545

		EPOCH: 17 LR: 0.004937187988281249

		Epoch: 17
		 [================================================================>]  Step: 156ms | Tot: 24s880ms | Train >> Loss: 0.247 | Acc: 91.302% (45651/50000) 98/98 
		 [=============================================================>...]  Step: 34ms | Tot: 1s445ms | Test >> Loss: 0.386 | Acc: 87.680% (8768/10000) 20/20 
		0.0007710643649101257

		EPOCH: 18 LR: 0.004261301757812499

		Epoch: 18
		 [================================================================>]  Step: 158ms | Tot: 24s864ms | Train >> Loss: 0.226 | Acc: 92.152% (46076/50000) 98/98 
		 [=============================================================>...]  Step: 34ms | Tot: 1s441ms | Test >> Loss: 0.341 | Acc: 88.920% (8892/10000) 20/20 
		0.000682312086224556

		EPOCH: 19 LR: 0.0035854155273437483

		Epoch: 19
		 [================================================================>]  Step: 159ms | Tot: 24s921ms | Train >> Loss: 0.212 | Acc: 92.672% (46336/50000) 98/98 
		 [=============================================================>...]  Step: 33ms | Tot: 1s585ms | Test >> Loss: 0.355 | Acc: 88.990% (8899/10000) 20/20 
		0.0007091604679822922

		EPOCH: 20 LR: 0.0029095292968749995

		Epoch: 20
		 [================================================================>]  Step: 156ms | Tot: 25s42ms | Train >> Loss: 0.195 | Acc: 93.204% (46602/50000) 98/98 
		 [=============================================================>...]  Step: 36ms | Tot: 1s563ms | Test >> Loss: 0.350 | Acc: 88.830% (8883/10000) 20/20 
		0.0007002256184816361

		EPOCH: 21 LR: 0.002233643066406249

		Epoch: 21
		 [================================================================>]  Step: 158ms | Tot: 24s862ms | Train >> Loss: 0.173 | Acc: 93.944% (46972/50000) 98/98 
		 [=============================================================>...]  Step: 35ms | Tot: 1s374ms | Test >> Loss: 0.329 | Acc: 89.830% (8983/10000) 20/20 
		0.0006588922172784805

		EPOCH: 22 LR: 0.0015577568359374985

		Epoch: 22
		 [================================================================>]  Step: 156ms | Tot: 24s869ms | Train >> Loss: 0.158 | Acc: 94.608% (47304/50000) 98/98 
		 [=============================================================>...]  Step: 34ms | Tot: 1s573ms | Test >> Loss: 0.314 | Acc: 89.970% (8997/10000) 20/20 
		0.0006274753004312516

		EPOCH: 23 LR: 0.0008818706054687497

		Epoch: 23
		 [================================================================>]  Step: 155ms | Tot: 25s48ms | Train >> Loss: 0.145 | Acc: 95.116% (47558/50000) 98/98 
		 [=============================================================>...]  Step: 34ms | Tot: 1s405ms | Test >> Loss: 0.309 | Acc: 90.370% (9037/10000) 20/20 
		0.000618566320836544

		EPOCH: 24 LR: 0.00020598437499999914

		Epoch: 24
		 [================================================================>]  Step: 160ms | Tot: 24s703ms | Train >> Loss: 0.132 | Acc: 95.548% (47774/50000) 98/98 
		 [=============================================================>...]  Step: 35ms | Tot: 1s370ms | Test >> Loss: 0.299 | Acc: 90.350% (9035/10000) 20/20 
		0.0005979749128222465

    ```

#**Analysis:**

- Used One Cycle Policy such that:
  - Total Epochs = 24
  - Max at Epoch = 5
  - LRMIN = (LRMAX/8) = 0.001647875
  - LRMAX = 0.013183
  - NO Annihilation
  - Transformations used:
  ```
  Transform: Compose(
               RandomCrop(size=(32, 32), padding=4)
               RandomHorizontalFlip(p=0.5)
               ToTensor()
               <lib.cutout.Cutout object at 0x7fe12a470668>
               Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201))
           )
   ```
  
  
- Created model as specified in S11
	- Cutout : Yes
	- Normalization:
	  - transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

	  



 
