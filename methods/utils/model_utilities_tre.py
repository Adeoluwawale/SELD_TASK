
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchaudio
from torchaudio.models import cnn14
class FrequencyTransfer(nn.Module):

    def __init__(
        self, 
        in_channels = 6, 
        output = 14, 
        model_name = 'Cnn14', 
        pretrained = False
        ):

        super(FrequencyTransfer, self).__init__()
        
        self.in_channels = in_channels
        self.output = output
        self.model_name = model_name
        self.pretrained = pretrained

        #self.m = torchaudio.models(self.model_name, pretrained=self.pretrained, num_classes=output)
       # def forward(self,x):
        
           # out=self.m(x)

           # return out
def change_first_layer(m):
  for conv_block1, child in m.named_children():
    if isinstance(child, nn.Conv2d):
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
m_sed = torchaudio.models.CNN14('Cnn14', pretrained=False, num_classes=13)
change_first_layer(m_sed)
#print(m)
#print(m(x).shape)
def forward(self,x):
        
            out=self.m(x)

            return out


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