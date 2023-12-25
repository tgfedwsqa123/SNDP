# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
#sys.path.append("..")
from cmath import nan
import numpy as np
import math
import os,sys,random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from DataSet_construct import generate_dataset

from models.model import SCNN_Model

device = torch.device("cuda:3")

def build_model(seed, A=0):

    seed_value = seed
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  
    torch.manual_seed(seed_value)                   
    torch.cuda.manual_seed(seed_value)              
    torch.cuda.manual_seed_all(seed_value)          
    torch.backends.cudnn.deterministic = True

    hyper_parameter = np.load("./record4_8class/32neuron_8class.npy")
    model = SCNN_Model(device,hyper_parameter,16)
    model.to(device)

    #print('model established')
    #print(model)
    model = torch.load("record2_4class/record_{}_0/32neuron_model.pth".format(seed))
    return model

def train(model, device, train_loader, optimizer, scheduler, criterion, epoch, A, check=False):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):  
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output,_ = model(data,A)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        if (batch_idx + 1) % 30 == 0 and not check:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), 50000,
                       100. * batch_idx / len(train_loader), loss.item()))
    acc = 100. * correct / 50000
    if not check:
        print("acc:",acc)
    scheduler.step()

def test(model, device, test_loader, criterion, epoch, A, check=False):
    test_loss = 0 
    correct = 0 
    with torch.no_grad():
        for data, target in test_loader: 
            data, target = data.to(device), target.to(device) 
            output,_ = model(data,0.03*A)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / 10000
    test_loss /= 10000
    if not check:
        print('\nTest set: Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            epoch, test_loss, correct, len(test_loader.dataset), acc))
    return acc
          
def main(model,seed,A,check=False):

    BATCH_SIZE = 256
    #Epoch = 15
    Epoch = 1

    train_loader,test_loader,val_loader = generate_dataset(BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5) #for fc model
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    bestacc = 0
    for e in range(Epoch):
        #train(model, device, train_loader, optimizer, scheduler, criterion, e, A, check)
        #acc = test(model, device, val_loader, criterion, e, A, check)
        acc = 100
        if acc > bestacc:
            bestacc = acc
            best_model = model
            save = 0
            if save:
                np.save("./record4_8class/record_{}_{}/32neuron_model_param.npy".format(seed,A),
                    {"conv1":np.array(model.conv1.weight.data.cpu()),
                    "conv1_b":np.array(model.conv1.bias.data.cpu()),
                    "conv2":np.array(model.conv2.weight.data.cpu()),
                    "conv2_b":np.array(model.conv2.bias.data.cpu()),
                    "fc1":np.array(model.fc1.weight.data.cpu()),
                    "fc1_b":np.array(model.fc1.bias.data.cpu()),
                    "fc2":np.array(model.fc2.weight.data.cpu()),
                    "fc2_b":np.array(model.fc2.bias.data.cpu())})
                torch.save(model,"record4_8class/record_{}_{}/32neuron_model.pth".format(seed,A))
        if not check:
            print("best_acc:", bestacc) 
    print("best_acc:", bestacc) 
    model = best_model
    acc = test(model, device, test_loader, criterion, Epoch, A, check)
    print("test_acc:",acc)      
    return acc

acc_dict = []
for seed in [0,1,2,3,4,5]:
    for A in [1,2,3,4,5]:
        print("seed:",seed,"A:",A)
        model = build_model(seed,A)
        acc = main(model,seed,A,check=False)
        acc_dict.append(acc)
print(acc_dict)