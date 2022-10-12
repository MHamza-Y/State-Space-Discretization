import torch
from torch import nn

from state_quantization.activations import StraightThroughEstimator
from state_quantization.forcasting_models import LSTMForcasting


class ForcastingDiscFinalState(LSTMForcasting):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_lstm_layer_activation = StraightThroughEstimator()
        self.quantized_state = []

    def forward(self, x):
        outputs = []
        (h, c) = self.init_hidden(x.shape[0])

        for i in range(self.seq_len):
            self.lstm_layers_forward(x=x[:, i, :], h=h, c=c)

        self.quantized_state = self.final_lstm_layer_activation(h[-1])
        output = self.final_dense_forward(self.quantized_state)

        outputs += [output.unsqueeze(1)]
        for t in range(self.look_ahead):
            x_t = x[:, self.seq_len + t, :].clone()

            x_t[:, self.replace_start:self.replace_end] = output

            self.lstm_layers_forward(x=x_t, h=h, c=c)
            qs = self.final_lstm_layer_activation(h[-1])
            output = self.final_dense_forward(qs)
            outputs += [output.unsqueeze(1)]

        outputs = torch.cat(outputs, dim=1)
        return outputs


class ForcastingDiscHC(LSTMForcasting):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.h_quantization_layers, self.c_quantization_layers = self._create_quantization_layers()
        self.quantized_state = []

    def _create_quantization_layers(self):
        h_layers = nn.ModuleList()
        c_layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            h_layers.append(StraightThroughEstimator())
            c_layers.append(StraightThroughEstimator())
        return h_layers, c_layers

    def lstm_layers_forward(self, x, h, c):
        layer_input = x
        for layer_idx in range(self.n_layers):
            (h[layer_idx], c[layer_idx]) = self.lstm_layers[layer_idx](layer_input, (h[layer_idx], c[layer_idx]))
            h[layer_idx] = self.h_quantization_layers[layer_idx](h[layer_idx])
            c[layer_idx] = self.c_quantization_layers[layer_idx](c[layer_idx])
            if layer_idx < self.n_layers - 1:
                h[layer_idx] = self.lstm_dropout_layers[layer_idx](h[layer_idx])
            layer_input = h[layer_idx]

    def forward(self, x):
        outputs = []
        (h, c) = self.init_hidden(x.shape[0])

        for i in range(self.seq_len):
            self.lstm_layers_forward(x=x[:, i, :], h=h, c=c)

        self.quantized_state = torch.cat((h[-1], c[-1]), dim=1)

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


class ForcastingDiscHCConst(ForcastingDiscHC):

    def forward(self, x):
        outputs = []
        (h, c) = self.init_hidden(x.shape[0])

        for i in range(self.seq_len):
            self.lstm_layers_forward(x=x[:, i, :], h=h, c=c)

        self.quantized_state = torch.cat((h[-1], c[-1]), dim=1)

        output = self.final_dense_forward(h[-1])

        outputs += [output.unsqueeze(1)]
        for t in range(self.look_ahead):
            outputs += [output.unsqueeze(1)]

        outputs = torch.cat(outputs, dim=1)
        return outputs


class ForcastingDiscFinalStateConst(ForcastingDiscFinalState):

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


class DiscAutoEncoder(nn.ModuleList):
    def __init__(self, input_size, bottleneck_size, encoder_hidden_shape=None, decoder_hidden_shape=None, dropout=0.2):
        super().__init__()
        if encoder_hidden_shape is None:
            encoder_hidden_shape = [input_size, input_size // 2]
        if decoder_hidden_shape is None:
            decoder_hidden_shape = encoder_hidden_shape[::-1]

        print(encoder_hidden_shape)
        print(decoder_hidden_shape)
        self.encoder_hidden_shape = encoder_hidden_shape
        self.decoder_hidden_shape = decoder_hidden_shape
        self.input_size = input_size
        self.bottleneck_size = bottleneck_size
        self.dropout = dropout

        self.layers = self.build_autoencoder_layers()

        self.encoder_layers, self.bottleneck_layers, self.decoder_layers = self.build_autoencoder_layers()
        self.bottleneck_out = []

        print('Encoder Layers')
        print(self.encoder_layers)
        print('Bottleneck Layers')
        print(self.bottleneck_layers)
        print('Decoded Layers')
        print(self.decoder_layers)

    def build_autoencoder_layers(self):

        encoder_layers = nn.ModuleList()
        decoder_layers = nn.ModuleList()
        bottleneck_layers = nn.ModuleList()
        last_out = self.input_size
        for enc_layer in self.encoder_hidden_shape:
            encoder_layers.append(nn.Linear(in_features=last_out, out_features=enc_layer))
            encoder_layers.append(nn.GELU())
            encoder_layers.append(nn.Dropout(self.dropout))
            last_out = enc_layer

        bottleneck_layers.append(nn.Linear(in_features=last_out, out_features=self.bottleneck_size))
        bottleneck_layers.append(StraightThroughEstimator())
        last_out = self.bottleneck_size

        for dec_layer in self.decoder_hidden_shape:
            decoder_layers.append(nn.Linear(in_features=last_out, out_features=dec_layer))
            decoder_layers.append(nn.GELU())
            decoder_layers.append(nn.Dropout(self.dropout))
            last_out = dec_layer

        decoder_layers.append(nn.Linear(in_features=last_out, out_features=self.input_size))

        return encoder_layers, bottleneck_layers, decoder_layers

    def forward(self, x):

        enc_out = x
        for enc_layer in self.encoder_layers:
            enc_out = enc_layer(enc_out)

        b_out = enc_out
        for b_layer in self.bottleneck_layers:
            b_out = b_layer(b_out)

        self.bottleneck_out = b_out
        dec_out = b_out
        for dec_layer in self.decoder_layers:
            dec_out = dec_layer(dec_out)

        return dec_out
