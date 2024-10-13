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
        rnn_hidden_size: int,
        bidirectional: bool = False,
        **batch
    ):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = rnn_hidden_size
        self.bidirectional = bidirectional

        self.num_directions = 2 if bidirectional else 1
        self.rnn = nn.LSTM(
            input_size,
            rnn_hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if bidirectional:
            self.down_projection = nn.Linear(rnn_hidden_size * 2, rnn_hidden_size)

    def forward(self, x, lengths):
        batch_size = x.size(0)
        batch_layer = nn.BatchNorm1d(num_features=batch_size)
        outputs = nn.ReLU(batch_layer(x.transpose(1, 2)).transpose(1, 2))
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.rnn(outputs)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs


class DeepSpeechModel(BaselineModel):
    # def __init__(self, input_channels, n_tokens, **batch):
    def __init__(self, input_channels, n_tokens, **batch):

        rnn_params = {"num_rnn": 5, "rnn_hidden_size": 512, "bidirectional": True}
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
        ]
        super().__init__(input_channels, n_tokens, **batch)

        self.input_channels = input_channels
        self.n_tokens = n_tokens
        self.conv_params = conv_params

        convs = []
        rnn_feat = input_channels
        for layer_ in self.conv_params:
            (out_channels, kernel_size) = layer_["out_channels"],  layer_["kernel_size"] 
            in_channels = layer_["in_channels"]
            stride = layer_['stride']
            rnn_feat = rnn_feat // stride[0]
            convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                        padding=tuple(np.array(kernel_size) // 2),
                        stride=tuple(stride),
                          bias=False),
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
