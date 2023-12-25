# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
#sys.path.append("..")
from cmath import nan
import numpy as np
import ray
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import cma
from torchvision import datasets, transforms
from DataSet_construct import generate_dataset

from models.model import SCNN_Model
from k_cluster_error import get_min_loss_kmeans

def train(model, device, train_loader, optimizer, scheduler, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output,_ = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
    scheduler.step()
def test(model, device, test_loader, criterion, epoch):
    test_loss = 0 
    correct = 0 
    with torch.no_grad():
        for data, target in test_loader: 
            data, target = data.to(device), target.to(device) 
            output, spiking_rate = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / 10000
    test_loss /= 10000
    return acc, spiking_rate
          
BATCH_SIZE = 256

train_loader,test_loader,val_loader = generate_dataset(BATCH_SIZE)

criterion = nn.CrossEntropyLoss()

class function():
    def __init__(self):
        super(function,self).__init__()
        self.Epoch = 15
        
    def f_cluster(self, hyper_parameter, neuron_number, p_number, device):
        hyper_parameter = abs(hyper_parameter)
        cluster_loss, k = get_min_loss_kmeans(hyper_parameter, neuron_number, p_number)
        return cluster_loss, k

    @ray.remote
    def f_cluster_cma(self, hyper_parameter, neuron_number, p_number, sigma, seed, device):
        mu = 20
        population = 30
        es = cma.CMAEvolutionStrategy(hyper_parameter, sigma, dict(maxiter=100, seed = seed, CMA_mu=mu, popsize = population)) 
        while not es.stop():
            solutions = es.ask()
            #for i in range(len(solutions)):
            #    solutions[i] = solutions[i].clip(min = 0.1)
            result = [self.f_cluster(x,neuron_number,p_number,device) for x in solutions]
            cluster_loss = [r[0] for r in result]
            k = [r[1] for r in result]
            es.tell(solutions, np.array(cluster_loss))
        x,_1,_2 = es.best.get()
        cluster_loss,k = self.f_cluster(x,neuron_number,p_number,device)
        return x, cluster_loss, k
        
    #@ray.remote
    def f_train(self, hyper_parameter, neuron_number, device):
        print(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
        hyper_parameter = abs(hyper_parameter)
        model = SCNN_Model(device,hyper_parameter,int(neuron_number/2)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5) #for fc model
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)
        bestacc = 0
        spiking_rate = torch.zeros(neuron_number,1).to(device)
        for e in range(self.Epoch):
            train(model, device, train_loader, optimizer, scheduler, criterion, e)
            acc, spiking_rate_tmp = test(model, device, val_loader, criterion, e)
            if acc > bestacc:
                bestacc = acc
                spiking_rate = spiking_rate_tmp
        reward = - bestacc
        print("best_acc:",bestacc)
        return reward, spiking_rate