import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from pylops import TorchOperator
from problem import forward, gradient
from unet2d import ResUNet

    
class Learned(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_max, Op):
        super(Learned, self).__init__()
        self.time_max = time_max
        self.net = ResUNet(input_size, output_size, hidden_channels=hidden_size, levels=2)
        self.Op = Op
        
    def forward(self, xt, y):
        inp = torch.cat([xt, 
                         gradient(xt.ravel(),
                                  self.Op, 
                                  y.ravel()).reshape(xt.shape)], 
                        0).reshape(1, 2, *xt.shape[1:])
        out = self.net.forward(inp).squeeze(1)
        return out
   
    def forward_sequence(self, x0, y, x, criterion): 
        xt = x0.clone()
        
        list_xt = [x0.squeeze().detach().numpy()]
        for t in range(self.time_max):
            dx = self.forward(xt, y)
            xt = xt + dx
            list_xt.append(xt.squeeze().detach().numpy())
        loss = criterion(xt.squeeze(), x.squeeze())
        list_xt = np.array(list_xt)
        
        return list_xt, loss
        
    def train_batch(self, X_batch, Y_batch, X0_batch, criterion):
        batch_loss = 0
        for i in range(len(X_batch)):
            x, y, x0 = X_batch[i], Y_batch[i], X0_batch[i]
            list_xt, loss = self.forward_sequence(x0, y, x, criterion)
            batch_loss += loss
        
        batch_loss /= len(X_batch)
        return batch_loss
    