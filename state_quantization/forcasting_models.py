import torch
from torch import nn


class LSTMForcasting(nn.Module):
    def __init__(self, features, hidden_size, out_size, seq_len, look_ahead, dropout=0.1, n_layers=1, replace_start=4,
                 replace_end=6, fully_connected_layers_shape=None):
        super().__init__()

        if fully_connected_layers_shape is None:
            fully_connected_layers_shape = [hidden_size, hidden_size]
        self.fully_connected_layers_shape = fully_connected_layers_shape
        self.seq_len = seq_len

        self.hidden_size = hidden_size
        self.out_size = out_size
        self.features = features

        self.n_layers = n_layers
        self.dropout = dropout
        self.look_ahead = look_ahead
        self.replace_end = replace_end
        self.replace_start = replace_start
        self.lstm_layers = self._create_lstm_cell_layers()
        self.lstm_dropout_layers = self._create_lstm_dropout_layers()
        self.fully_connected_layers = self._create_fully_connected_layers()
        self.hidden_states = []

        print('LSTM Layers')
        print(self.lstm_layers)
        print('LSTM Dropout Layers')
        print(self.lstm_dropout_layers)
        print('Fully Connected Layers')
        print(self.fully_connected_layers)

    def _create_lstm_cell_layers(self):
        layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):

            if layer_idx == 0:
                layer_input_size = self.features
            else:
                layer_input_size = self.hidden_size
            layers.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=self.hidden_size))

        return layers

    def _create_lstm_dropout_layers(self):
        layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            if layer_idx < self.n_layers - 1:
                layers.append(nn.Dropout(self.dropout))
        return layers

    def _create_fully_connected_layers(self):
        layers = nn.ModuleList()
        last_out = self.hidden_size
        for layer_shape in self.fully_connected_layers_shape:
            layers.append(nn.Linear(in_features=last_out, out_features=layer_shape))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout))
            last_out = layer_shape

        layers.append(nn.Linear(in_features=last_out, out_features=self.out_size))

        layers = nn.Sequential(*layers)

        return layers

    def lstm_layers_forward(self, x, h, c):
        layer_input = x
        for layer_idx in range(self.n_layers):
            (h[layer_idx], c[layer_idx]) = self.lstm_layers[layer_idx](layer_input, (h[layer_idx], c[layer_idx]))
            if layer_idx < self.n_layers - 1:
                h[layer_idx] = self.lstm_dropout_layers[layer_idx](h[layer_idx])
            layer_input = h[layer_idx]

    def final_dense_forward(self, x):
        return self.fully_connected_layers(x)

    def forward(self, x):
        outputs = []
        (h, c) = self.init_hidden(x.shape[0])

        for i in range(self.seq_len):
            self.lstm_layers_forward(x=x[:, i, :], h=h, c=c)

        self.hidden_states = torch.cat((h[-1], c[-1]), dim=1)
        output = self.final_dense_forward(h[-1])

        outputs += [output.unsqueeze(1)]
        for t in range(self.look_ahead):
            x_t = x[:, self.seq_len + t, :].clone()

            x_t[:, self.replace_start:self.replace_end] = output

            self.lstm_layers_forward(x=x_t, h=h, c=c)
            output = self.final_dense_forward(h[-1])
            outputs += [output.unsqueeze(1)]

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def set_look_ahead(self, look_ahead):
        self.look_ahead = look_ahead

    def get_seq_len(self):
        return self.seq_len

    def get_device(self):
        return next(self.parameters()).device

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def init_hidden(self, batch_size):
        device = self.get_device()
        return ([torch.zeros(batch_size, self.hidden_size).to(device) for i in range(self.n_layers)],
                [torch.zeros(batch_size, self.hidden_size).to(device) for i in range(self.n_layers)])


class LSTMForcastingConstOut(LSTMForcasting):

    def forward(self, x):
        outputs = []
        (h, c) = self.init_hidden(x.shape[0])

        for i in range(self.seq_len):
            self.lstm_layers_forward(x=x[:, i, :], h=h, c=c)

        self.hidden_states = torch.cat((h[-1], c[-1]), dim=1)
        output = self.final_dense_forward(h[-1])

        outputs += [output.unsqueeze(1)]
        for t in range(self.look_ahead):
            outputs += [output.unsqueeze(1)]

        outputs = torch.cat(outputs, dim=1)
        return outputs
