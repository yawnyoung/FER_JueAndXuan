"""
Sequential facial expression recognition

author: Yajue Yang
"""

import torch.nn as nn
import torch.nn.functional as F


class SFER_LSTM(nn.Module):
    def __init__(self):
        super(SFER_LSTM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

    def forward(self, seq):

        len_seq = seq.size()[0]

        # cnn for each frame in the sequence
        for frame_idx in range(len_seq):
            x = self.conv1(seq[frame_idx])
            x = F.relu(x)
            x = self.conv2(x)

        # take the sequence of features as input to the following LSTM layers
