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

    def __init__(self, features, hidden_size, out_size, seq_len, look_ahead, dropout_p=0.1, n_layers=1,
                 lstm_activation=None, replace_start=4, replace_end=6):
        super().__init__()
        self.seq_len = seq_len

        self.hidden_size = hidden_size
        self.out_size = out_size
        self.features = features

        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.look_ahead = look_ahead
        self.replace_end = replace_end
        self.replace_start = replace_start

        self.lstm_cells = self._create_lstm_cell_layers()
        self.final_lstm_layer_activation = StraightThroughEstimator() if lstm_activation is None else lstm_activation
        self.dropout_layers = self._create_dropout_layers()
        self.fcl1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fcl1_act = nn.GELU()
        self.fcl2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fcl2_act = nn.GELU()
        self.fcl3 = nn.Linear(in_features=hidden_size, out_features=out_size)

    def _create_lstm_cell_layers(self):
        layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):

            if layer_idx == 0:
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
            if layer_idx == 0:
                layer_input = x
            else:
                layer_input = h[layer_idx - 1]
            (h[layer_idx], c[layer_idx]) = self.lstm_cells[layer_idx](layer_input, (h[layer_idx], c[layer_idx]))
            if layer_idx < self.n_layers - 1:
                h[layer_idx] = self.dropout_layers[layer_idx](h[layer_idx])

    def final_dense_forward(self, h):
        activated_out = self.final_lstm_layer_activation(h[-1])
        fcl1_out = self.fcl1(activated_out)
        fcl1_act_out = self.fcl1_act(fcl1_out)
        fcl2_out = self.fcl2(fcl1_act_out)
        fcl2_act_out = self.fcl2_act(fcl2_out)
        fcl3_out = self.fcl3(fcl2_act_out)
        return fcl3_out, activated_out

    def forward(self, x):
        outputs = []
        (h, c) = self.init_hidden(x.shape[0])

        for i in range(self.seq_len):
            self.lstm_layers_forward(x=x[:, i, :], h=h, c=c)

        output, self.quantized_state = self.final_dense_forward(h)

        outputs += [output.unsqueeze(1)]
        for t in range(self.look_ahead):
            x_t = x[:, self.seq_len + t, :].clone()

            x_t[:, self.replace_start:self.replace_end] = output

            self.lstm_layers_forward(x=x_t, h=h, c=c)
            output, _ = self.final_dense_forward(h)
            outputs += [output.unsqueeze(1)]

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def get_device(self):
        return next(self.parameters()).device

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def init_hidden(self, batch_size):
        device = self.get_device()
        # Initial (h, c) for multilayer lstm
        return ([torch.zeros(batch_size, self.hidden_size).to(device) for i in range(self.n_layers)],
                [torch.zeros(batch_size, self.hidden_size).to(device) for i in range(self.n_layers)])
        # return (torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(device),
        #         torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(device))


class ConstLookAheadModel(DynamicsLookAheadModel):

    def forward(self, x):
        outputs = []
        (h, c) = self.init_hidden(x.shape[0])

        for i in range(self.seq_len):
            self.lstm_layers_forward(x=x[:, i, :], h=h, c=c)

        output, self.quantized_state = self.final_dense_forward(h)

        outputs += [output.unsqueeze(1)]
        for t in range(self.look_ahead):
            outputs += [output.unsqueeze(1)]

        outputs = torch.cat(outputs, dim=1)
        return outputs
