import torch
import torch.nn as nn
import torchvision.models as models
import torchaudio
import torch.nn.functional as F
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
    def __init__(self, in_channels, out_channels,):
        
        super(ConvBlock, self, ).__init__()
        
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
class Cnn14(nn.Module):
     def __init__(self, classes_num): # sample_rate, window_size, hop_size, mel_bins, fmin, fmax
       
        
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # # Spectrogram extractor
        # self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
        #     win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
        #     freeze_parameters=True)

        # # Logmel feature extractor
        # self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
        #     n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
        #     freeze_parameters=True)

        # # Spec augmenter
        # self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
        #     freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

        def init_weight(self):
            init_bn(self.bn0)
            init_layer(self.fc1)
            init_layer(self.fc_audioset)
    
    
 
        def forward(self, input, mixup_lambda=None):
            """
            Input: (batch_size, data_length)"""

        x = (input)   # (batch_size, 1, time_steps, freq_bins)#self.spectrogram_extractor(input) 
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict

class Transfer_Cnn14(nn.Module):
      def __init__(self, classes_num):
        super(Transfer_Cnn14, self).__init__()
        self.base = Cnn14(classes_num)
      def change_first_layer(m,self):
            for ConvBlock, child in m.named_children():
                if isinstance(child, nn.Conv2D):
                 kwargs = {
                   'out_channels': child.out_channels,
                   'kernel_size': child.kernel_size,
                    'stride': child.stride,
                    'padding': child.padding,
                    'bias': False if child.bias == None else True
                }
                 m._modules[self.conv_block1] = nn.Conv2d(4, **kwargs)#[name]
                 return True
            else:
              if(change_first_layer(child)):
                   return True
            return False

#x = torch.randn((8, 15, 224, 224))
m_sed = torchaudio.models('Cnn14', pretrained=False, num_classes=13)
change_first_layer(m_sed)

class Transfer_Cnn14(nn.Module):
      def __init__(self, classes_num,):
        super(Transfer_Cnn14, self).__init__()
        self.base = Cnn14(classes_num)
      def change_first_layer(m,self):
            for conv_block1, child in m.named_children():
                if isinstance(child, nn.Conv2D):
                 kwargs = {
                   'out_channels': child.out_channels,
                   'kernel_size': child.kernel_size,
                    'stride': child.stride,
                    'padding': child.padding,
                    'bias': False if child.bias == None else True
                }
                 m._modules[self.conv_block1] = nn.Conv2d(7, **kwargs)#[name]
                 return True
            else:
              if(change_first_layer(child)):
                   return True
            return False



#x = torch.randn((8, 15, 224, 224))
m_doa = torchaudio.models('Cnn14', pretrained=False, num_classes=14)
change_first_layer(m_doa)

class Transfer_Cnn14(nn.Module):
      def __init__(self, classes_num,):
        super(Transfer_Cnn14, self).__init__()
        self.base = Cnn14(classes_num)
      def change_first_layer(m,self):
            for conv_block1, child in m.named_children():
                if isinstance(child, nn.Conv2D):
                 kwargs = {
                   'out_channels': child.out_channels,
                   'kernel_size': child.kernel_size,
                    'stride': child.stride,
                    'padding': child.padding,
                    'bias': False if child.bias == None else True
                }
                 m._modules[self.conv_block1] = nn.Conv2d(7, **kwargs)#[name]
                 return True
            else:
              if(change_first_layer(child)):
                   return True
            return False

