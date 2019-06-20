import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, n_classes, in_channels):
        super(CRNN, self).__init__()
        self.conv_block = ConvBN(in_channels, out_channels=128, padding=3)
        self.gru = nn.GRU(input_size=128, hidden_size=64,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=128, out_features=n_classes)

    def forward(self, x):
        x = self.gru(self.conv_block(x).transpose(1, 2))[0]
        return torch.sigmoid(self.fc(x))


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, **args):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, **args)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
