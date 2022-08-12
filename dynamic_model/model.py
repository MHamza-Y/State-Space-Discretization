import torch
from torch import nn

from dynamic_model.activations import StraightThroughEstimator


class DynamicsModel(nn.Module):

    def __init__(self, features, hidden_size, out_size, seq_len, batch_size, dropout_p=0.5, n_layers=1,
                 lstm_activation=None):
        super().__init__()
        self.seq_len = seq_len

        self.hidden_size = hidden_size
        self.out_size = out_size
        self.features = features
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.lstm_cells = self._create_lstm_cell_layers()
        self.lstm_activation = StraightThroughEstimator() if lstm_activation is None else lstm_activation
        self.dropout_layers = self._create_dropout_layers()
        self.fcl = nn.Linear(in_features=hidden_size, out_features=out_size)

    def _create_lstm_cell_layers(self):
        layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):

            if layer_idx is 0:
                layer_input_size = self.features
            else:
                layer_input_size = self.hidden_size
            layers.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=self.hidden_size))
        return layers

    def _create_dropout_layers(self):
        dropout_layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            dropout_layers.append(nn.Dropout(p=self.dropout_p))
        return dropout_layers

    def forward(self, x):
        (h, c) = self.init_hidden()

        for t in range(self.seq_len):
            for layer_idx in range(self.n_layers):
                if layer_idx is 0:
                    layer_input = x[:, t, :]
                else:
                    layer_input = h[layer_idx - 1]

                (h[layer_idx], c[layer_idx]) = self.lstm_cells[layer_idx](layer_input, (h[layer_idx], c[layer_idx]))

                if layer_idx < self.n_layers - 1:
                    h[layer_idx] = self.dropout_layers[layer_idx](h[layer_idx])
        activated_out = self.lstm_activation(h[-1])
        output = self.fcl(activated_out)
        return output.unsqueeze(1)

    def init_hidden(self):
        device = next(self.parameters()).device
        # Initial (h, c) for multilayer lstm
        return ([torch.zeros(self.batch_size, self.hidden_size).to(device) for i in range(self.n_layers)],
                [torch.zeros(self.batch_size, self.hidden_size).to(device) for i in range(self.n_layers)])
        # return (torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(device),
        #         torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(device))


class DynamicsLookAheadModel(nn.Module):

    def __init__(self, features, hidden_size, out_size, seq_len, batch_size, look_ahead, dropout_p=0.5, n_layers=1,
                 lstm_activation=None, replace_start=0, replace_end=6):
        super().__init__()
        self.seq_len = seq_len

        self.hidden_size = hidden_size
        self.out_size = out_size
        self.features = features
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.look_ahead = look_ahead
        self.replace_end = replace_end
        self.replace_start = replace_start

        self.lstm_cells = self._create_lstm_cell_layers()
        self.lstm_activation = StraightThroughEstimator() if lstm_activation is None else lstm_activation
        self.dropout_layers = self._create_dropout_layers()
        self.fcl = nn.Linear(in_features=hidden_size, out_features=out_size)

    def _create_lstm_cell_layers(self):
        layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):

            if layer_idx is 0:
                layer_input_size = self.features
            else:
                layer_input_size = self.hidden_size
            layers.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=self.hidden_size))
        return layers

    def _create_dropout_layers(self):
        dropout_layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            dropout_layers.append(nn.Dropout(p=self.dropout_p))
        return dropout_layers

    def lstm_layers_forward(self, x, h, c):
        for layer_idx in range(self.n_layers):
            if layer_idx is 0:
                layer_input = x
            else:
                layer_input = h[layer_idx - 1]
            (h[layer_idx], c[layer_idx]) = self.lstm_cells[layer_idx](layer_input, (h[layer_idx], c[layer_idx]))
            if layer_idx < self.n_layers - 1:
                h[layer_idx] = self.dropout_layers[layer_idx](h[layer_idx])

    def final_dense_forward(self, h):
        activated_out = self.lstm_activation(h[-1])
        output = self.fcl(activated_out)
        return output

    def forward(self, x):
        outputs = []
        (h, c) = self.init_hidden()

        for i in range(self.seq_len):
            self.lstm_layers_forward(x=x[:, i, :], h=h, c=c)

        output = self.final_dense_forward(h)

        outputs += [output.unsqueeze(1)]
        for t in range(self.look_ahead):
            x_t = x[:, t, :].clone().detach()

            x_t[:, self.replace_start:self.replace_end] = output
            x_t = x_t.requires_grad_()

            self.lstm_layers_forward(x=x_t, h=h, c=c)
            output = self.final_dense_forward(h)
            outputs += [output.unsqueeze(1)]

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def init_hidden(self):
        device = next(self.parameters()).device
        # Initial (h, c) for multilayer lstm
        return ([torch.zeros(self.batch_size, self.hidden_size).to(device) for i in range(self.n_layers)],
                [torch.zeros(self.batch_size, self.hidden_size).to(device) for i in range(self.n_layers)])
        # return (torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(device),
        #         torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(device))
