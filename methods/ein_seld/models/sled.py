from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.utils.model_utilities_tre import  ( FrequencyModel,PositionalEncoding,  init_layer) #( CustomCNN,  OrthogonalConv2d,PositionalEncoding, DoubleCNN, init_layer)
                                          




class EINV2(nn.Module):
    def __init__(self, cfg, dataset):
        super().__init__()
        self.pe_enable = False  # Ture | False

        if cfg['data']['audio_feature'] == 'logmel&intensity':
            self.f_bins = cfg['data']['n_mels']
            self.in_channels = 7
          
        
        self.downsample_ratio = 2 ** 2   #sed_conv_block1
        self.m_sed = nn.Sequential(
           FrequencyModel(),
            nn.AvgPool2d(kernel_size=(2, 2)),

        )
        self.doa_conv_block1 = nn.Sequential(
           FrequencyModel(conv_block1= self.in_channels),
            nn.AvgPool2d(kernel_size=(2, 2)),

        )
        if self.pe_enable:
            self.pe = PositionalEncoding(pos_len=100, d_model=512, pe_type='t', dropout=0.0)
        self.sed_trans_track1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)
        self.sed_trans_track2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)
        self.doa_trans_track1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)
        self.doa_trans_track2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)

        self.fc_sed_track1 = nn.Linear(512, 14, bias=True)
        self.fc_sed_track2 = nn.Linear(512, 14, bias=True)
        self.fc_doa_track1 = nn.Linear(512, 3, bias=True)
        self.fc_doa_track2 = nn.Linear(512, 3, bias=True)
        self.final_act_sed = nn.Sequential() # nn.Sigmoid()
        self.final_act_doa = nn.Tanh()

        self.init_weight()
    
    def init_weight(self):

        init_layer(self.fc_sed_track1)
        init_layer(self.fc_sed_track2)
        init_layer(self.fc_doa_track1)
        init_layer(self.fc_doa_track2)
    
    
    def forward(self, x):
        """
        x: waveform, (batch_size, num_channels, data_length)
        """
        x_sed = x[:, :4]
        x_doa = x
       # transformer
        if self.pe_enable:
            x_sed = self.pe(x_sed)
        if self.pe_enable:
            x_doa = self.pe(x_doa)
        x_sed = x_sed.permute(2, 0, 1) # (T, N, C)
        x_doa = x_doa.permute(2, 0, 1) # (T, N, C)

        x_sed_1 = self.sed_trans_track1(x_sed).transpose(0, 1) # (N, T, C)
        x_sed_2 = self.sed_trans_track2(x_sed).transpose(0, 1) # (N, T, C)   
        x_doa_1 = self.doa_trans_track1(x_doa).transpose(0, 1) # (N, T, C)
        x_doa_2 = self.doa_trans_track2(x_doa).transpose(0, 1) # (N, T, C)

        # fc
        x_sed_1 = self.final_act_sed(self.fc_sed_track1(x_sed_1))
        x_sed_2 = self.final_act_sed(self.fc_sed_track2(x_sed_2))
        x_sed = torch.stack((x_sed_1, x_sed_2), 2)
        x_doa_1 = self.final_act_doa(self.fc_doa_track1(x_doa_1))
        x_doa_2 = self.final_act_doa(self.fc_doa_track2(x_doa_2))
        x_doa = torch.stack((x_doa_1, x_doa_2), 2)
        output = {
            'sed': x_sed,
            'doa': x_doa,
        }

        return output  