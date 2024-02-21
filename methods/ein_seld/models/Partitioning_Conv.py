import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Example usage:
input_channels = 3
output_channels = 64
model = DoubleCNN(input_channels, output_channels)
