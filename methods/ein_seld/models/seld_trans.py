from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from methods.utils.model_utilities_transfer  import (Transfer_Cnn14, PositionalEncoding,  init_layer) #( CustomCNN,  OrthogonalConv2d,PositionalEncoding, DoubleCNN, init_layer)
import torchaudio                                         
from methods.utils.transfer_doa import(Transfer_Cnn14_d)

class EINV2(nn.Module):
    def __init__(self, cfg, dataset):
        super().__init__()
        self.pe_enable = False  # Ture | False
        self.in_channels= 4
        self.in_channels_doa = 7
        freeze_base = False
        if cfg['data']['audio_feature'] == 'logmel&intensity':
            self.f_bins = cfg['data']['n_mels']
            # self.in_channels_doa  = 7
            # self.in_channels_sed = 4
        
        self.downsample_ratio = 2 ** 2
        self.sed = nn.Sequential(
            Transfer_Cnn14(in_channels = 4,  classes_num = 14, freeze_base = False), #nn.AvgPool2d(kernel_size=(2, 2)
              #nn.AvgPool2d(2, 2)

        )
        # self.sed = (Transfer_Cnn14(4,  classes_num = 14, freeze_base = False),
        #       nn.AvgPool2d(kernel_size=(2, 2))
        # )
        self.doa= nn.Sequential(
             Transfer_Cnn14_d(in_channels = 7,  classes_num = 14, freeze_base = False),
              #nn.AvgPool2d(2, 2) 
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

        for param in Transfer_Cnn14.parameters(self):
            param.requires_grad = False
        if  freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

            self.init_weights()
        for param in Transfer_Cnn14_d.parameters(self):
            param.requires_grad = False
        if  freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

            self.init_weights()
    # def init_weights(self):
    #     init_layer(self) #.fc_transfer

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load('/mnt/raid/ni/WALE_SEdl/EIN-SELD/Cnn14_DecisionLevelMax_mAP=0.385.pth') # pretrained_checkpoint_path
        self.base.load_state_dict(checkpoint['model']) #model

    def forward(self, input,mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']
    def init_weight(self):

        init_layer(self.fc_sed_track1)
        init_layer(self.fc_sed_track2)
        init_layer(self.fc_doa_track1)
        init_layer(self.fc_doa_track2)
    
    
    def forward(self, x):
        """
        x: waveform, (batch_size, num_channels, data_length)
        """
        x_sed = x[:, :2048] #4
        x_doa = x
        
        x_sed = self.sed(x_sed)
        x_doa = self.doa(x_doa)
    #     x_sed = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 0], x_sed) + \
    #         torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 1], x_doa)
    #     x_doa = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 0], x_sed) + \
    #         torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 1], x_doa)
    #     # x_sed = self.sed_conv_block2(x_sed)
    #     # x_doa = self.doa_conv_block2(x_doa)
    #     # x_sed = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 0], x_sed) + \
    #     #     torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 1], x_doa)
    #     # x_doa = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 0], x_sed) + \
    #     #     torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 1], x_doa)
    #     # x_sed = self.sed_conv_block3(x_sed)
    #     # x_doa = self.doa_conv_block3(x_doa)
    #     # x_sed = torch.einsum('c, nctf -> nctf', self.stitch[2][:, 0, 0], x_sed) + \
    #     #     torch.einsum('c, nctf -> nctf', self.stitch[2][:, 0, 1], x_doa)
    #     # x_doa = torch.einsum('c, nctf -> nctf', self.stitch[2][:, 1, 0], x_sed) + \
    #     #     torch.einsum('c, nctf -> nctf', self.stitch[2][:, 1, 1], x_doa)
    #     # x_sed = self.sed_conv_block4(x_sed)
    #     # x_doa = self.doa_conv_block4(x_doa)
        # x_sed = x_sed.mean(dim=3) # (N, C, T)
        # x_doa = x_doa.mean(dim=3) # (N, C, T)
  
      
    #    # transformer
        # if self.pe_enable:
        #     x_sed = self.pe(x_sed)
        # if self.pe_enable:
        #     x_doa = self.pe(x_doa)
        # x_sed = x_sed.permute(2, 0, 1) # (T, N, C)
        # x_doa = x_doa.permute(2, 0, 1) # (T, N, C)

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