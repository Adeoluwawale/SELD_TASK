import math
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer(layer, nonlinearity='leaky_relu'):
    '''
    Initialize a layer
    '''
    classname = layer.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)

class OrthogonalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(OrthogonalConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.init_weights()

    def init_weights(self):
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)

class DoubleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleCNN, self).__init__()
        self.cnn1 = OrthogonalConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cnn2 = OrthogonalConv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        return x
class PositionalEncoding(nn.Module):
    def __init__(self, pos_len, d_model=512, pe_type='t', dropout=0.0):
        """ Positional encoding using sin and cos

        Args:
            pos_len: positional length
            d_model: number of feature maps
            pe_type: 't' | 'f' , time domain, frequency domain
            dropout: dropout probability
        """
        super().__init__()
        
        self.pe_type = pe_type
        pe = torch.zeros(pos_len, d_model)
        pos = torch.arange(0, pos_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = 0.1 * torch.sin(pos * div_term)
        pe[:, 1::2] = 0.1 * torch.cos(pos * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2) # (N, C, T)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # x is (N, C, T, F) or (N, C, T) or (N, C, F)
        if x.ndim == 4:
            if self.pe_type == 't':
                pe = self.pe.unsqueeze(3)
                x += pe[:, :, :x.shape[2]]
            elif self.pe_type == 'f':
                pe = self.pe.unsqueeze(2)
                x += pe[:, :, :, :x.shape[3]]
        elif x.ndim == 3:
            x += self.pe[:, :, :x.shape[2]]
        return self.dropout(x)

