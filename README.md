# EVA_4_Assignments

# Model Summary:

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
#================================================================
            Conv2d-1           [-1, 16, 26, 26]             160
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
           Dropout-5           [-1, 16, 26, 26]               0
           Dropout-6           [-1, 16, 26, 26]               0
           Dropout-7           [-1, 16, 26, 26]               0
           Dropout-8           [-1, 16, 26, 26]               0
           Dropout-9           [-1, 16, 26, 26]               0
          Dropout-10           [-1, 16, 26, 26]               0
           Conv2d-11           [-1, 32, 24, 24]           4,640
             ReLU-12           [-1, 32, 24, 24]               0
      BatchNorm2d-13           [-1, 32, 24, 24]              64
          Dropout-14           [-1, 32, 24, 24]               0
          Dropout-15           [-1, 32, 24, 24]               0
          Dropout-16           [-1, 32, 24, 24]               0
          Dropout-17           [-1, 32, 24, 24]               0
          Dropout-18           [-1, 32, 24, 24]               0
          Dropout-19           [-1, 32, 24, 24]               0
          Dropout-20           [-1, 32, 24, 24]               0
           Conv2d-21           [-1, 10, 24, 24]             330
        MaxPool2d-22           [-1, 10, 12, 12]               0
           Conv2d-23           [-1, 10, 10, 10]             910
             ReLU-24           [-1, 10, 10, 10]               0
      BatchNorm2d-25           [-1, 10, 10, 10]              20
          Dropout-26           [-1, 10, 10, 10]               0
          Dropout-27           [-1, 10, 10, 10]               0
          Dropout-28           [-1, 10, 10, 10]               0
          Dropout-29           [-1, 10, 10, 10]               0
          Dropout-30           [-1, 10, 10, 10]               0
          Dropout-31           [-1, 10, 10, 10]               0
          Dropout-32           [-1, 10, 10, 10]               0
           Conv2d-33             [-1, 16, 8, 8]           1,456
             ReLU-34             [-1, 16, 8, 8]               0
      BatchNorm2d-35             [-1, 16, 8, 8]              32
          Dropout-36             [-1, 16, 8, 8]               0
          Dropout-37             [-1, 16, 8, 8]               0
          Dropout-38             [-1, 16, 8, 8]               0
          Dropout-39             [-1, 16, 8, 8]               0
          Dropout-40             [-1, 16, 8, 8]               0
          Dropout-41             [-1, 16, 8, 8]               0
          Dropout-42             [-1, 16, 8, 8]               0
           Conv2d-43             [-1, 16, 6, 6]           2,320
             ReLU-44             [-1, 16, 6, 6]               0
      BatchNorm2d-45             [-1, 16, 6, 6]              32
          Dropout-46             [-1, 16, 6, 6]               0
          Dropout-47             [-1, 16, 6, 6]               0
          Dropout-48             [-1, 16, 6, 6]               0
          Dropout-49             [-1, 16, 6, 6]               0
          Dropout-50             [-1, 16, 6, 6]               0
          Dropout-51             [-1, 16, 6, 6]               0
          Dropout-52             [-1, 16, 6, 6]               0
           Conv2d-53             [-1, 16, 4, 4]           2,320
             ReLU-54             [-1, 16, 4, 4]               0
          Dropout-55             [-1, 16, 4, 4]               0
          Dropout-56             [-1, 16, 4, 4]               0
          Dropout-57             [-1, 16, 4, 4]               0
          Dropout-58             [-1, 16, 4, 4]               0
          Dropout-59             [-1, 16, 4, 4]               0
          Dropout-60             [-1, 16, 4, 4]               0
          Dropout-61             [-1, 16, 4, 4]               0
           Conv2d-62             [-1, 10, 1, 1]           2,570
          Dropout-63             [-1, 10, 1, 1]               0
          Dropout-64             [-1, 10, 1, 1]               0
          Dropout-65             [-1, 10, 1, 1]               0
          Dropout-66             [-1, 10, 1, 1]               0
          Dropout-67             [-1, 10, 1, 1]               0
          Dropout-68             [-1, 10, 1, 1]               0
          Dropout-69             [-1, 10, 1, 1]               0
#================================================================
Total params: 14,886
Trainable params: 14,886
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 2.50
Params size (MB): 0.06
Estimated Total Size (MB): 2.56
----------------------------------------------------------------


# Test Accuracy:

 0%|          | 0/469 [00:00<?, ?it/s]
##### Epoch Number : 1 ######
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:72: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.2603204846382141 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.27it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0798, Accuracy: 9760/10000 (98%)

##### Epoch Number : 2 ######
loss=0.2973267734050751 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.10it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0530, Accuracy: 9830/10000 (98%)

##### Epoch Number : 3 ######
loss=0.15657939016819 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.44it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0461, Accuracy: 9857/10000 (99%)

##### Epoch Number : 4 ######
loss=0.1879035085439682 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.31it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0364, Accuracy: 9881/10000 (99%)

##### Epoch Number : 5 ######
loss=0.06673335283994675 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 28.29it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0326, Accuracy: 9894/10000 (99%)

##### Epoch Number : 6 ######
loss=0.15884339809417725 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.56it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0309, Accuracy: 9895/10000 (99%)

##### Epoch Number : 7 ######
loss=0.14488841593265533 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.78it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0281, Accuracy: 9899/10000 (99%)

##### Epoch Number : 8 ######
loss=0.1598663479089737 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 27.65it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0294, Accuracy: 9899/10000 (99%)

##### Epoch Number : 9 ######
loss=0.09440817683935165 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 26.49it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0249, Accuracy: 9917/10000 (99%)

##### Epoch Number : 10 ######
loss=0.16562773287296295 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.17it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0320, Accuracy: 9899/10000 (99%)

##### Epoch Number : 11 ######
loss=0.17639313638210297 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 24.63it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0286, Accuracy: 9908/10000 (99%)

##### Epoch Number : 12 ######
loss=0.062057893723249435 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.28it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0253, Accuracy: 9916/10000 (99%)

##### Epoch Number : 13 ######
loss=0.042007844895124435 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.02it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0228, Accuracy: 9917/10000 (99%)

##### Epoch Number : 14 ######
loss=0.1397893875837326 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.48it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0246, Accuracy: 9926/10000 (99%)

##### Epoch Number : 15 ######
loss=0.058261264115571976 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.51it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0242, Accuracy: 9922/10000 (99%)

##### Epoch Number : 16 ######
loss=0.04986128583550453 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.53it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0203, Accuracy: 9933/10000 (99%)

##### Epoch Number : 17 ######
loss=0.12678025662899017 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.51it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0223, Accuracy: 9925/10000 (99%)

##### Epoch Number : 18 ######
loss=0.12406963855028152 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.25it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0213, Accuracy: 9934/10000 (99%)

##### Epoch Number : 19 ######
loss=0.07693588733673096 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 29.90it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0257, Accuracy: 9923/10000 (99%)

##### Epoch Number : 20 ######
loss=0.07674974948167801 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.94it/s]

Test set: Average loss: 0.0214, Accuracy: 9931/10000 (99%)
