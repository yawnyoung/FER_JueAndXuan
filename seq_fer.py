"""
Sequential facial expression recognition

author: Yajue Yang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from seq_fer_datasets import SAMPLE_INPUT, SAMPLE_TARGET


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


def pad_tensor_batch(tensor_list, tensor_lens):
    """
    Pad a batch of tensors with variable lengths
    :param tensor_list: [t1, t2, ..., tN]
    :param tensor_lens: a list of tensor lengths
    :return:
    """
    # find longest length
    max_len = max(tensor_lens)

    # argument sort (descending order)
    sorted_idx = np.argsort(-np.asarray(tensor_lens))

    paddes_list = list(map(lambda idx: pad_tensor(tensor_list[idx], max_len, 0), sorted_idx))

    # print(torch.stack(paddes_list).size())

    return torch.stack(paddes_list), [tensor_lens[idx] for idx in sorted_idx], sorted_idx


class SFER_LSTM(nn.Module):

    def __init__(self):
        super(SFER_LSTM, self).__init__()

        self.num_classes = 8

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 9 * 9, out_features=500)
        self.pool = nn.MaxPool2d(2, 2)

        self.lstm1 = nn.LSTM(500, 250, batch_first=True)
        self.lstm2 = nn.LSTM(250, 120, batch_first=True)

        self.fc2 = nn.Linear(in_features=120, out_features=self.num_classes)

    def pad_tensor_batch(self, tensor_list, tensor_lens):
        """
        Pad a batch of tensors with variable lengths
        :param tensor_list: [t1, t2, ..., tN]
        :param tensor_lens: a list of tensor lengths
        :return:
        """
        # find longest length
        max_len = max(tensor_lens)

        # argument sort (descending order)
        sorted_idx = np.argsort(-np.asarray(tensor_lens))

        paddes_list = list(map(lambda idx: pad_tensor(tensor_list[idx].data, max_len, 0), sorted_idx))

        # print(torch.stack(paddes_list).size())

        return torch.stack(paddes_list), [tensor_lens[idx] for idx in sorted_idx], sorted_idx

    def conv_fcs(self, img_batch, seq_len):
        """
        convolutional layers to extract image features
        :param img_batch: a batch of all images in the input videos
        :param seq_len: lengths of each sequence
        :return: list of features
        """
        x = self.conv1(img_batch)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # print(x.size())

        x = x.view(-1, 16 * 9 * 9)

        x = self.fc1(x)
        x = F.relu(x)

        feat_batch = []
        seq_start = 0
        for s_idx in range(len(seq_len)):
            seq_end = seq_start + seq_len[s_idx]
            feat_batch.append(x[seq_start:seq_end])
            seq_start = seq_end

        # for feat in feat_batch:
        #     print(feat.size())

        return feat_batch

    def forward(self, seq_batch):
        """
        forward function
        :param seq_batch: batch of sequences ([s1, s2, ..., sN], [l1, l2, ..., lN])
        :return:
        """

        batch_size = len(seq_batch[0])

        seq_len = [len(seq) for seq in seq_batch[0]]
        # print('lengths of seqs: ', seq_len)

        # stack images
        all_images = Variable(torch.cat(seq_batch[0], dim=0))
        # print(all_images.size())

        # convolution layers
        img_feats = self.conv_fcs(all_images, seq_len)

        # # pad image features with variable lengths
        padded_img_feats, sorted_seq_len, sorted_idx = self.pad_tensor_batch(img_feats, seq_len)
        padded_img_feats = Variable(padded_img_feats)
        # print('padded img feats: ', padded_img_feats.size())
        packed_pad_feats = nn.utils.rnn.pack_padded_sequence(padded_img_feats, sorted_seq_len, batch_first=True)
        # print('packed feats:', packed_pad_feats.data.size())
        # print('packed batch size: ', packed_pad_feats.batch_sizes)

        # lstm layers
        x, hc1 = self.lstm1(packed_pad_feats)
        x, hc2 = self.lstm2(x)
        # print(hc2[0].size())

        # output
        output, ret_lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # print(output.size())
        # print(ret_lens)

        last_frame_output = []
        for b_idx in range(batch_size):
            # print(ret_lens[b_idx])
            last_frame_output.append(output[b_idx, ret_lens[b_idx]-1, :])

        last_frame_output = torch.stack(last_frame_output)
        # print(last_frame_output.size())

        # fc layers
        out = self.fc2(last_frame_output)

        return out, sorted_idx


class Simple_LSTM(nn.Module):
    def __init__(self):
        super(Simple_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(1 * 48 * 48, 1000, batch_first=True)
        self.lstm2 = nn.LSTM(1000, 500, batch_first=True)
        self.fc1 = nn.Linear(500, 250)
        self.fc2 = nn.Linear(250, 8)

    def forward(self, x):

        batch_size = len(x[0])

        seq_len = [len(v) for v in x[0]]

        # pad videos
        expanded_video = [v.view(v.size()[0], -1) for v in x[0]]
        padded_img, sorted_seq_len, sorted_idx = pad_tensor_batch(expanded_video, seq_len)
        padded_img = Variable(padded_img)
        packed_img = nn.utils.rnn.pack_padded_sequence(padded_img, sorted_seq_len, batch_first=True)

        out, hc1 = self.lstm1(packed_img)
        out, hc2 = self.lstm2(out)

        out, ret_lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # last frame output
        last_frame_output = []
        for b_idx in range(batch_size):
            last_frame_output.append(out[b_idx, ret_lens[b_idx]-1, :])

        last_frame_output = torch.stack(last_frame_output)

        # fc layers
        out = self.fc1(last_frame_output)
        out = F.relu(out)

        out = self.fc2(out)

        return out, sorted_idx
