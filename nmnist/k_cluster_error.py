from sklearn.cluster import KMeans
import numpy as np
import torch
import math

def cos(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def distance(a,b):
    return np.linalg.norm(a-b)

def get_k_loss_kmeans(data,neuron_number,n):
    estimator = KMeans(n_clusters=n)
    estimator.fit(data)
    label = estimator.labels_
    total_loss = 0
    
    for i in range(n):
        cluster_dict = []
        for j in range(neuron_number):
            if label[j] == i:
                cluster_dict.append(data[j,:])
        number = len(cluster_dict)
        center = estimator.cluster_centers_[i]
        loss = 0
        for j in range(number):
            loss += distance(cluster_dict[j], center)
        total_loss += loss
    return total_loss + n * 0.05

def get_min_loss_kmeans(data,neuron_number,p_number):
    
    data_reshape = torch.zeros(neuron_number,p_number)
    for i in range(p_number):
        data_reshape[:,i] = torch.tensor(data[i*neuron_number:(i+1)*neuron_number])

    k = 2
    min_loss = 0
    number = int(neuron_number / k)
    for i in range(k):
        min_loss += get_k_loss_kmeans(data_reshape[i*number:(i+1)*number,:], number, 1)

    return min_loss, k