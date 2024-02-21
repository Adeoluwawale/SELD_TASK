import torch
import torch.nn as nn

import torchaudio

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_utils import do_mixup, interpolate, pad_framewise_output
 

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

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x



class FrequencyModel(nn.Module):

    def __init__(
        self, 
        in_channels = 6, 
        output = 9, 
        model_name = 'resnet200d', 
        pretrained = False
        ):

        super(FrequencyModel, self).__init__()
        
        self.in_channels = in_channels
        self.output = output
        self.model_name = model_name
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.output = output
        self.model_name = model_name
        self.pretrained = pretrained

        self.m = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=output)

    def change_first_layer(m):
        for name, child in m.named_children():
              if isinstance(child,nn.Conv2d ): #nn.Conv2d
                kwargs = {
                'out_channels': child.out_channels,
                'kernel_size': child.kernel_size,
                'stride': child.stride,
                'padding': child.padding,
                'bias': False if child.bias == None else True
       }
              m._modules[conv_block1] = nn.Conv2d(4, **kwargs)
              return True
        else:
            if(change_first_layer(child)):
             return True
        return False

#x = torch.randn((8, 15, 224, 224))
m = torchaudio.create_model('Cnn14', pretrained=False, num_classes=9)
change_first_layer(m)
print(m)
#print(m(x).shape)