import random,os
import numpy as np
import torch
#from spikingjelly.datasets import pad_sequence_collate, padded_sequence_mask
from spikingjelly.datasets.n_mnist import NMNIST
from torch.utils.data import DataLoader, SubsetRandomSampler

seed_value = 1   
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  
torch.manual_seed(seed_value)                   
torch.cuda.manual_seed(seed_value)              
torch.cuda.manual_seed_all(seed_value)          
torch.backends.cudnn.deterministic = True

def generate_dataset(batch_size):
    indices = np.load("./data/indices.npy")
    nmnist_train = NMNIST(root='./data/N-MNIST', train=True, data_type='frame', frames_number=20, split_by='number')
    nmnist_test = NMNIST(root='./data/N-MNIST', train=False, data_type='frame', frames_number=20, split_by='number')
    train_indices = indices[0:50000]
    val_indices = indices[50000:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_data_loader = DataLoader(dataset=nmnist_train, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, sampler=train_sampler)
    val_data_loader = DataLoader(dataset=nmnist_train, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, sampler=val_sampler)
    test_data_loader = DataLoader(dataset=nmnist_test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    return train_data_loader, test_data_loader, val_data_loader




