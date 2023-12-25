import torch
import torch.nn as nn
import numpy as np
from thop import profile
import pickle
from importlib import import_module
from models.model import SCNN_Model
device = torch.device("cuda:0")
inputs = torch.randn(1,2,34,34,dtype=torch.float).to(device)
big_model = SCNN_Model(device,torch.rand(64,1),16).to(device)
small_model = SCNN_Model(device,torch.rand(8,1),2).to(device)
print("big:")
flops, params = profile(big_model, (inputs,0))
print('flops: ', flops)
print('params: ', params)

print("small:")
flops, params = profile(small_model, (inputs,0))
print('flops: ', flops)
print('params: ', params)