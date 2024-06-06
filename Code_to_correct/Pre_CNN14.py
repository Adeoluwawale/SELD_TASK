from torch.nn.modules.batchnorm import BatchNorm2d
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
# pretrained_model_path = "path_to_pretrained_model.pth"
# pretrained_model = torch.load(pretrained_model_path)
class Cnn14(nn.Module):
     def __init__( self, in_channels, out_channels, pretrained_weights_path=None):
          
          super(Cnn14, self).__init__()

          self.in_channels = in_channels
          self.out_channels = out_channels
          self.bn0 = nn.BatchNorm2d(64) #64
         
          self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
          self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
          self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
          self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
          self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
          self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)  
          if pretrained_weights_path is not None:
              pretrained_state_dict = torch.load(pretrained_weights_path)
              self.load_state_dict(pretrained_state_dict, strict=False)  
     def init_weight(self):
        init_bn(self.bn0)
        
     def forward(self, input, ):
         """
         Input: (batch_size, data_length)"""
         x = (input)
         x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
         x = F.dropout(x, p=0.2)
         x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
         x = F.dropout(x, p=0.2)
         x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
         x = F.dropout(x, p=0.2)
         x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
         x = F.dropout(x, p=0.2)
         x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
         x = F.dropout(x, p=0.2)
         x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
         x = F.dropout(x, p=0.2)
         x = torch.mean(x, dim=3)
         (x1, _) = torch.max(x, dim=2)
         x2 = torch.mean(x, dim=2)
         x = x1 + x2
         x = F.dropout(x, p=0.5)

         return x

#modify your layers here
Cnn14.conv_block1= ConvBlock(in_channels= 7, out_channels= 64)


def forward(self, x):
       
        x = self.conv_block1(x)

        return x









# pretrained_model_path = '/mnt/raid/ni/WALE_SEdl/EIN-SELD/Cnn14_DecisionLevelMax_mAP=0.385.pth'
# model = Cnn14(num_classes=14, pretrained_weights_path= pretrained_weights_path)
# model.conv_block1= ConvBlock(in_channels= 4, out_channels= 64)
