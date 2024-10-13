import logging

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np

from src.model.baseline_model import BaselineModel


class RNNLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bidirectional: bool = False,
    ):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.num_directions = 2 if bidirectional else 1
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if bidirectional:
            self.down_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, lengths):
        batch_size = x.size(0)
        batch_layer = nn.BatchNorm1d(num_features=batch_size)
        outputs = nn.ReLU(batch_layer(x.transpose(1, 2)).transpose(1, 2))
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.rnn(outputs)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs


class DeepSpeechModel(BaselineModel):
    def __init__(self, input_channels, n_tokens,
        rnn_params = {"num_rnn": 5,  "rnn_hid": 512, "bi": True},
        conv_params = [
  {
    "in_channels": 1,
    "out_channels": 32,
    "kernel_size": [41, 11],
    "stride": [2, 2]
  },
  {
    "in_channels": 32,
    "out_channels": 32,
    "kernel_size": [21, 11],
    "stride": [2, 1]
  }
], **batch):
        super().__init__(input_channels, n_tokens, **batch)
        self.input_channels = input_channels
        self.n_tokens = n_tokens
        self.conv_params = conv_params

        convs = []
        rnn_feat = input_channels
        for out_channels, kernel_size in self.conv_params:
            convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels=out_channels, kernel_size=kernel_size, padding=tuple(np.array(kernel_size) // 2), bias=False),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.Hardtanh(0, 20)
                )
            )
        self.conv = nn.Sequential(*convs)

        rnn_hidden_size = (1 + rnn_params['bidirectional']) * rnn_params['rnn_hidden_size']
        self.rnn = nn.ModuleList([RNNLayer(input_size=rnn_feat, **rnn_params)] +
                                 [RNNLayer(input_size=rnn_hidden_size, **rnn_params) for _ in range(rnn_params['num_rnn'] - 1)])

        self.fc = nn.Linear(in_features=rnn_feat, out_features=n_tokens)

    def forward(self, x, spectrogram_length,  **batch):
        outputs = x

        outputs = self.conv(outputs.unsqueeze(1))
        outputs = outputs.reshape(outputs.shape[0], outputs.shape[1] * outputs.shape[2], outputs.shape[3])
        outputs = outputs.transpose(1, 2)

        for conv_layer in self.conv_params:
            spectrogram_length = (spectrogram_length.subtract(1).float() / conv_layer['stride'][1]).add(1).int()

        for rnn_layer in self.rnn:
            outputs = rnn_layer(outputs, spectrogram_length)

        return {"logits": self.fc(outputs)}
