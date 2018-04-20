"""
Sequential facial expression recognition

author: Yajue Yang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SFER_LSTM(nn.Module):
    def __init__(self):
        super(SFER_LSTM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

    def forward(self, seq):

        len_seq = seq[0].size()[1]
        print(len_seq)

        # cnn for each frame in the sequence
        seq_feature = []
        for frame_idx in range(len_seq):
            x = self.conv1(Variable(seq[0][:, frame_idx, :, :, :]))
            x = F.relu(x)
            x = self.conv2(x)
            seq_feature.append(x)

        # seq_feature = torch.stack(seq_feature)
        # print(seq_feature.size())
        # take the sequence of features as input to the following LSTM layers
