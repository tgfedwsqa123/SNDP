import torch
import torch.nn as nn
import numpy as np
import pickle
from importlib import import_module
from models.model_measure_spikes import SCNN_Model
from DataSet_construct import generate_dataset

def test(model, device, test_loader):
   correct_train = 0 
   with torch.no_grad():
      for data, target in test_loader: 
         target = target.to(device)
         data = data.to(device)
         output,spikes = model(data,0)
         _,predict = torch.max(output,dim=1)
         correct_train += predict.eq(target.view_as(predict)).sum().item()
   acc = 100. * correct_train / len(test_loader.dataset)
   print('\nTest set:Accuracy: {}/{} ({:.1f}%)\n'.format(
      correct_train, len(test_loader.dataset),  acc))
   return spikes

device = torch.device("cuda:3")

hyper_parameter_origin = np.load("./record4_8class/record_0_0/32neuron_8class.npy")
hyper_parameter_origin = np.array(abs(hyper_parameter_origin))
neuron_number = 8
scale = int(32/neuron_number)
hyper_parameter = np.zeros((neuron_number*2))
for i in range(neuron_number*2):
   hyper_parameter[i] = np.mean(hyper_parameter_origin[scale*i:scale*(i+1)])
big_model = SCNN_Model(device,hyper_parameter_origin,16).to(device)
small_model = SCNN_Model(device,hyper_parameter,int(neuron_number/2)).to(device)
model_big = (torch.load("record4_8class/record_1_0/32neuron_model.pth"))
model_small = (torch.load("record4_8class/record_0_0/8neuron_model_ri.pth"))
big_model.load_state_dict(model_big.state_dict())
small_model.load_state_dict(model_small.state_dict())
if 1:
   small_model.thr_conv1 = model_small.thr_conv1
   small_model.thr_conv2 = model_small.thr_conv2
   small_model.decay_conv1 = model_small.decay_conv1
   small_model.decay_conv2 = model_small.decay_conv2
   small_model.to(device)

BATCH_SIZE = 512

train_loader,test_loader,val_loader = generate_dataset(BATCH_SIZE)
spike_big = test(big_model,device,test_loader)
print("big:",spike_big)
spike_small = test(small_model,device,test_loader)
print("small:",spike_small)