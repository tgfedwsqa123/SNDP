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

device = torch.device("cuda:0")

def build_model(seed, need_init=True, A=0):

    seed_value = seed
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  
    torch.manual_seed(seed_value)                   
    torch.cuda.manual_seed(seed_value)              
    torch.cuda.manual_seed_all(seed_value)          
    torch.backends.cudnn.deterministic = True
    #print('model established')
    hyper_parameter_origin = np.load("./record2_4class/record_{}_{}/32neuron_4class.npy".format(seed,A))
    hyper_parameter_origin = np.array(abs(hyper_parameter_origin))
    neuron_number = 4
    scale = int(32/neuron_number)
    hyper_parameter = np.zeros((neuron_number*2))
    for i in range(neuron_number*2):
        hyper_parameter[i] = np.mean(hyper_parameter_origin[scale*i:scale*(i+1)])

    #print(hyper_parameter)
    if need_init and 0:
        model = SCNN_Model(device,hyper_parameter,int(neuron_number/2))
    else:
        model = SCNN_Model(device,torch.rand_like(torch.tensor(hyper_parameter)),int(neuron_number/2))
    model.to(device)
    #print(model)

    weight_data = np.load("./record2_4class/record_{}_{}/32neuron_model_param.npy".format(seed,A),allow_pickle=True)
    weight_data = weight_data.tolist()
    conv1 = weight_data["conv1"]
    conv1_b = weight_data["conv1_b"]
    conv2 = weight_data["conv2"]
    conv2_b = weight_data["conv2_b"]
    fc1 = weight_data["fc1"]
    fc1_b = weight_data["fc1_b"]
    fc2 = weight_data["fc2"]
    fc2_b = weight_data["fc2_b"]

    conv2_tmp = np.zeros((16,int(neuron_number/2),3,3))
    for i in range(int(neuron_number/2)):
        conv2_tmp[:,i,...] = np.sum(conv2[:,scale*i:scale*(i+1),...],axis=1)
    conv2 = conv2_tmp

    if need_init:
        for i in range(int(neuron_number/2)):
            model.conv1.weight.data[i,...] = torch.tensor(np.mean(conv1[scale*i:scale*(i+1),...],axis=0),device=device)
            model.conv1.bias.data[i] = torch.tensor(np.mean(conv1_b[scale*i:scale*(i+1)]),device=device)
        for i in range(int(neuron_number/2)):
            model.conv2.weight.data[i,...] = torch.tensor(np.mean(conv2[scale*i:scale*(i+1),...],axis=0),device=device)
            model.conv2.bias.data[i] = torch.tensor(np.mean(conv2_b[scale*i:scale*(i+1)]),device=device)
        for i in range(32):
            model.fc1.weight.data[:,i] = torch.tensor(np.sum(fc1[:,scale*i:scale*(i+1)],axis=1),device=device)
        model.fc1.bias.data = torch.tensor(fc1_b,device=device)
        model.fc2.weight.data = torch.tensor(fc2,device=device)
        model.fc2.bias.data = torch.tensor(fc2_b,device=device)
    #model = torch.load("record2_4class/record_{}_0/4neuron_model_ri.pth".format(seed))
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
                       100. * batch_idx / 50000, loss.item()))
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
            epoch, test_loss, correct, 10000, acc))
    return acc
          
def main(model,seed,A,check=False):

    BATCH_SIZE = 256
    Epoch = 20
    #Epoch = 1

    train_loader,test_loader,val_loader = generate_dataset(BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1e-4) #for fc model
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    bestacc = 0
    for e in range(Epoch):
        train(model, device, train_loader, optimizer, scheduler, criterion, e, A, check)
        acc = test(model, device, val_loader, criterion, e, A, check)
        if acc > bestacc:
            bestacc = acc
            best_model = model
            save = 1
            if save:
                np.save("./record2_4class/record_{}_{}/4neuron_model_param_withweight.npy".format(seed,A),
                    {"conv1":np.array(model.conv1.weight.data.cpu()),
                    "conv1_b":np.array(model.conv1.bias.data.cpu()),
                    "conv2":np.array(model.conv2.weight.data.cpu()),
                    "conv2_b":np.array(model.conv2.bias.data.cpu()),
                    "fc1":np.array(model.fc1.weight.data.cpu()),
                    "fc1_b":np.array(model.fc1.bias.data.cpu()),
                    "fc2":np.array(model.fc2.weight.data.cpu()),
                    "fc2_b":np.array(model.fc2.bias.data.cpu())})
                torch.save(model,"record2_4class/record_{}_{}/4neuron_model_withweight.pth".format(seed,A))
        if not check:
            print("best_acc:", bestacc) 
    print("best_acc:", bestacc) 
    model = best_model
    acc = test(model, device, test_loader, criterion, Epoch, A, check)
    print("test_acc:",acc)   
    return acc 

acc_dict = []
for seed in [0,1,2,3,4,5]:
    for A in [0]:
        print("seed:",seed,"A:",A)
        model = build_model(seed,need_init=True,A=A)
        acc = main(model,seed,A,check=False)
        acc_dict.append(acc)
print(acc_dict)