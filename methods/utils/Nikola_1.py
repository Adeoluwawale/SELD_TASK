import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer(layer, nonlinearity='leaky_relu'):
    '''
    Initialize a layer
    '''
    classname = layer.__class__.__name__
    if (classname.find('CNN') != -1) or (classname.find('Linear') != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)

class CustomCNN(nn.Module):
    def __init__(self,  in_channels, out_channels, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                dilation=1, bias=False):
        super().__init__()
       # super(CustomCNN, self).__init__()
        
        self.Conv_dou= nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation = dilation,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size,stride=stride,
                              padding= padding, dilation = dilation, bias=bias)
        )                       
        #self.bn1 = nn.BatchNorm2d(out_channels)
        #self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()
        
   
    def init_weights(self):
        for layer in self.Conv_dou:
            init_layer(layer)

        
    def forward(self, x):
        x = self.Conv_dou(x)

        return x

        
    # Function to perform SVD and split filters into two orthogonal sub-spaces
def split_filters_orthogonal(conv_layer):
    weight = conv_layer.weight
    in_channels, out_channels, kernel_size, _ = weight.size()

    # Reshape the weight tensor to perform SVD
    weight = weight.view(in_channels, out_channels * kernel_size * kernel_size)

    # Perform SVD
    U, S, V = torch.svd(weight, some=False)

    # Split into two orthogonal sub-spaces
    sub_space1 = U[:, :out_channels]
    sub_space2 = U[:, out_channels:]

    return sub_space1, sub_space2

# # Initialize your CNN model
# model = CustomCNN(in_channels =3, out_channels= 64)

# # Iterate through the blocks and apply SVD
# for layer in [model.block1, model.block2]:
#     for layer in  layer :#block:
#         if isinstance(layer, nn.Conv2d):
#             sub_space1, sub_space2 = split_filters_orthogonal(layer)

#             # Create new convolutional layers with the decomposed weights
#             new_layer1 = nn.Conv2d(layer.in_channels, sub_space1.size(1), kernel_size=layer.kernel_size, padding=layer.padding[0])
#             new_layer2 = nn.Conv2d(layer.in_channels, sub_space2.size(1), kernel_size=layer.kernel_size, padding=layer.padding[0])

#             # Initialize the new layers with the decomposed weights
#             new_layer1.weight.data = sub_space1.view(new_layer1.weight.size())
#             new_layer2.weight.data = sub_space2.view(new_layer2.weight.size())

#             # Replace the old layer with the new layers
#             layer = nn.Sequential(new_layer1, new_layer2)
            
# # Print the updated model
# print(model)

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

