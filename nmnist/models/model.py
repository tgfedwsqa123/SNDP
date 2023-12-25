import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

lens = 0.5

# 激活
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply

class SCNN_Model(nn.Module):
    def __init__(self, device, hyper_parameter=None, channel_number=8):
        super(SCNN_Model, self).__init__()
        
        self.channel_number = channel_number
        self.device = device
        self.fc_hidden = 32
        self.decay = 0.3
        self.thr_conv1 = torch.rand(channel_number, requires_grad=False).to(device)
        self.thr_conv2 = torch.rand(channel_number, requires_grad=False).to(device)
        self.decay_conv1 = torch.rand(channel_number, requires_grad=False).to(device)
        self.decay_conv2 = torch.rand(channel_number, requires_grad=False).to(device)
        
        if hyper_parameter is not None:
            hyper_parameter = abs(hyper_parameter)
            self.thr_conv1 = torch.tensor(hyper_parameter[0:channel_number], requires_grad=False).to(device)
            self.thr_conv2 = torch.tensor(hyper_parameter[channel_number:2*channel_number], requires_grad=False).to(device)
            
            self.decay_conv1 = torch.tensor(hyper_parameter[2*channel_number:3*channel_number], requires_grad=False).to(device)
            self.decay_conv2 = torch.tensor(hyper_parameter[3*channel_number:4*channel_number], requires_grad=False).to(device)
            
        self.conv1 = nn.Conv2d(2, channel_number, kernel_size=3, stride=1, padding=1) # 2*28*28 -> 32*28*28
        self.conv2 = nn.Conv2d(channel_number, channel_number, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(channel_number * 8 * 8, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, 10)

    def mem_update(self, ops, x, mem, spike, thr, decay, v_reset):
        act_fun = ActFun.apply
        mem = mem * decay * (1. - spike) + spike * v_reset + ops(x)
        spike = act_fun(mem-thr) # act_fun : approximation firing function
        return mem, spike

    def forward(self, input, A):
        batch_size = input.shape[0]
        time_window = input.shape[1]
        noise = torch.stack([torch.torch.tensor(A * np.load("models/noise.npy"))]*batch_size,dim=0).to(self.device)
        input = input + noise

        c1_mem = c1_spike = torch.zeros(batch_size, self.channel_number, 34, 34, device=self.device)
        c2_mem = c2_spike = torch.zeros(batch_size, self.channel_number, 17, 17, device=self.device)
         
        ####################### DOUBLE CHECK ##########################################
        ###############################################################################
        thr_conv1 = torch.zeros_like(c1_mem, device=self.device, requires_grad=False)
        thr_conv2 = torch.zeros_like(c2_mem, device=self.device, requires_grad=False)
        
        decay_conv1 = torch.zeros_like(c1_mem, device=self.device, requires_grad=False)
        decay_conv2 = torch.zeros_like(c2_mem, device=self.device, requires_grad=False)
        
        for i in range(self.channel_number):
            thr_conv1[:,i,:,:] = self.thr_conv1[i]
            thr_conv2[:,i,:,:] = self.thr_conv2[i]
            decay_conv1[:,i,:,:] = self.decay_conv1[i]
            decay_conv2[:,i,:,:] = self.decay_conv2[i]
        #################################################################################
        
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, self.fc_hidden, device=self.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, 10, device=self.device)

        for step in range(time_window): # simulation time steps
            x = input[:,step,:,:,:].to(self.device)
            
            c1_mem, c1_spike = self.mem_update(self.conv1, x.float(), c1_mem, c1_spike, thr_conv1, decay_conv1, 0)
            x = F.avg_pool2d(c1_spike, 2)
            c2_mem, c2_spike = self.mem_update(self.conv2, x,         c2_mem, c2_spike, thr_conv2, decay_conv2, 0)
            x = F.avg_pool2d(c2_spike, 2)
            x = x.view(batch_size, -1)

            h1_mem, h1_spike = self.mem_update(self.fc1, x,        h1_mem, h1_spike, 0.5, self.decay, 0)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = self.mem_update(self.fc2, h1_spike, h2_mem, h2_spike, 0.5, self.decay, 0)
            h2_sumspike += h2_spike
        
        outputs = h2_sumspike / time_window
        return outputs, 0
