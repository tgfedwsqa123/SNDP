from gettext import find
from functions import function
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import os, random
import time
import cma
import ray
import seaborn as sns
import heapq

seed_value = 1   
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  
torch.manual_seed(seed_value)                   
torch.cuda.manual_seed(seed_value)              
torch.cuda.manual_seed_all(seed_value)          
torch.backends.cudnn.deterministic = True
func = function()
neuron_number = 16

mu = 10
population = 20
iteration = 1
device = torch.device("cuda:3")

def save_param_plot(reward,cluster_loss,k,param,iter):
    parameter = np.zeros((neuron_number,2))
    for i in range(2):
        parameter[:,i] = param[i*neuron_number:(i+1)*neuron_number]
    parameter = abs(parameter)
    fig = plt.figure(tight_layout=True)
    gs = matplotlib.gridspec.GridSpec(1, 3)
    fig.add_subplot(gs[0, 0:])
    sns.heatmap(parameter[:,:])
    plt.savefig("./param_picture/{}_{}_{}_{}.jpg".format(iter,round(reward,2),round(cluster_loss,2),k))
    plt.close("all")

init_param = np.array(torch.rand(neuron_number * 2)).tolist()
iteration = 1
best_solution = np.array(init_param)
sigma1 = 0.5 * pow(0.5,iteration-1)
best_reward = 0
best_spiking_rate = torch.zeros((neuron_number,1))
best_acc = 0
for i in range(20):
    sigma1 = sigma1 * 0.5
    start_time = time.time()
    if sigma1 > 0.001:
        result = ray.get([func.f_cluster_cma.remote(func,best_solution,neuron_number,2,sigma1,seed,device) for seed in range(population)])
        solutions = [r[0] for r in result]
        cluster_loss = [r[1] for r in result]
        k = [r[2] for r in result]
        print(cluster_loss)
        best_index_dict = list(map(cluster_loss.index, heapq.nsmallest(population, cluster_loss)))
        reward = np.zeros((population))
        result = [func.f_train(solutions[best_index_dict[i]],neuron_number,device) for i in range(20)]
        reward[best_index_dict] = [r[0] for r in result]
        coe = 1
        f_value = np.array(reward)+coe*np.array(cluster_loss)
        best_index = np.argmin(f_value)
        best_solution = solutions[best_index]
        best_reward = reward[best_index]
        print("reward:",best_reward)
        if best_reward < best_acc:
            best_acc = best_reward
            print("best:",best_acc)

    cluster_loss,k = func.f_cluster(best_solution,neuron_number,2,device)
    
    np.save("./best_param/16neuron_2class_{}.npy".format(iteration),np.array(best_solution))
    save_param_plot(best_reward,cluster_loss,k,best_solution,iteration)
    iteration += 1
    print(time.time() - start_time)