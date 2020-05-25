import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from utils.utils import progress_bar

def train(network, trainloader, device, optimizer, criterion, epoch):
    print('\nEpoch: %d' % epoch)
    network.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Train >> Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        '''print('Train:: Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))'''
        
def test(network, testloader, device, criterion, epoch):
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Test >> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            '''print('Test:: Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))'''
                
def testWithAccPlt(network, testloader, device, criterion, valaccuracies, vallosses, epoch):
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Test >> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), accuracy, correct, total))
            
            # scheduler.step(effective_loss)
    
    valaccuracies.append(accuracy)
    test_loss /= len(testloader.dataset)
    vallosses.append(test_loss)
    print(test_loss)
    return test_loss
    
def trainWithAccPlt(network, trainloader, device, optimizer, criterion, trainaccuracies, trainlosses, epoch):
    print('\nEpoch: %d' % epoch)
    network.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        accuracy = 100.*correct/total
        
        progress_bar(batch_idx, len(trainloader), 'Train >> Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), accuracy, correct, total))
        
        '''print('Train:: Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))'''
            
    trainaccuracies.append(accuracy)
    train_loss /= len(trainloader.dataset)
    trainlosses.append(train_loss)

def plotmetrics(target,trainaccuracies, testaccuracies, trainlosses, testlosses, savefilename):
    
    
    fig, axs = plt.subplots(1, 2, figsize=(15,10))
    
    # Plot Accuracy
    axs[0].plot(trainaccuracies, label='Train Accuracy')
    axs[0].plot(testaccuracies, label='Test Accuracy')
    axs[0].set_title("Accuracy")
    axs[0].legend(loc="upper left")
    
    # Plot loss
    axs[1].plot(trainlosses, label='Train Loss')
    axs[1].plot(testlosses, label='Test Loss')
    axs[1].set_title("Loss")
    axs[1].legend(loc="upper left")
    
    plt.show()
    fig.savefig("{}.png".format(savefilename))
    
    print('Max. Train Accuracy outoff {}-epochs is : {} at {}-Epoach'.format(len(trainaccuracies),max(trainaccuracies),trainaccuracies.index(max(trainaccuracies))+1))
    print('Max. Test Accuracy  outoff {}-epochs is : {} at {}-Epoach'.format(len(testaccuracies),max(testaccuracies),testaccuracies.index(max(testaccuracies))+1))
    
    print("\n\nMin. Train Loss outoff {}-epochs is : {:.6f} at {}-Epoach".format(len(trainlosses),min(trainlosses),trainlosses.index(min(trainlosses))+1))
    print("Min. Test Loss  outoff {}-epochs is : {:.6f} at {}-Epoach".format(len(testlosses),min(testlosses),testlosses.index(min(testlosses))+1))
    
    
    for i,v in enumerate(trainaccuracies):
        if v>float(target):
            print("\n\nTarget-{}% achieved at {}th epoach,Train accuracy is : {}".format(float(target),i+1,v))
            break
        
    for i,v in enumerate(testaccuracies):
        if v>float(target):
            print("Target-{}% achieved at {}th epoach,Test accuracy is : {}".format(float(target),i+1,v))
            break