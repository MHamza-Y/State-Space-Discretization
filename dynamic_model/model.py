import torch
from torch import nn


class DynamicsModel(nn.Module):

    def __init__(self, features, hidden_size, out_size, seq_len, batch_size, dropout_p=0.5):
        super().__init__()

        self.seq_len = seq_len

        self.hidden_size = hidden_size
        self.out_size = out_size
        self.features = features
        self.batch_size = batch_size

        self.lstm_cell = nn.LSTMCell(input_size=features, hidden_size=hidden_size)
        self.lstm_sig = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(p=dropout_p)
        self.fcl = nn.Linear(in_features=hidden_size, out_features=out_size)

    def forward(self, x, hc):
        for t in range(self.seq_len):
            hc = self.lstm_cell(x[:, t, :], hc)

        h, c = hc

        sig_out = self.lstm_sig(h)
        output = self.fcl(self.dropout_layer(sig_out))

        return output

    def init_hidden(self):
        device = next(self.parameters()).device
        return (torch.zeros(self.batch_size, self.hidden_size).to(device),
                torch.zeros(self.batch_size, self.hidden_size).to(device))
