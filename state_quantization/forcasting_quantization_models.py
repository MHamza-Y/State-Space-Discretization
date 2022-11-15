import torch
from torch import nn

from state_quantization.forcasting_models import LSTMForcasting
from state_quantization.quantization_models import DiscAutoEncoder


class ForcastingQuant(nn.Module):

    def __init__(self, forcasting_model: LSTMForcasting, autoencoder_quant_model: DiscAutoEncoder):
        super().__init__()
        self.forcasting_model = forcasting_model
        self.autoencoder_quant_model = autoencoder_quant_model
        self.quantized_state = []
        self.autoencoder_in = []

    def forward(self, x):
        forcasting_out = self.forcasting_model(x)
        self.autoencoder_in = self.forcasting_model.hidden_states.detach()
        autoencoder_out = self.autoencoder_quant_model(self.autoencoder_in)
        self.quantized_state = self.autoencoder_quant_model.bottleneck_out

        return forcasting_out, autoencoder_out

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def set_look_ahead(self, look_ahead):
        self.forcasting_model.look_ahead = look_ahead

    def get_seq_len(self):
        return self.forcasting_model.seq_len

    def get_device(self):
        return next(self.parameters()).device


class ForcastingQuantInferenceWrapper(nn.Module):

    def __init__(self, forcasting_quant_model: ForcastingQuant):
        super().__init__()
        self.model = forcasting_quant_model
        self.quantized_state = []

    def forward(self, x):
        forcasting_out, _ = self.model(x)
        self.quantized_state = self.model.quantized_state
        return forcasting_out

    def set_look_ahead(self, look_ahead):
        self.model.set_look_ahead(look_ahead)

    def get_device(self):
        return next(self.parameters()).device

    def get_seq_len(self):
        return self.model.get_seq_len()


class EmbeddedAEForcastingQuant(LSTMForcasting):

    def __init__(self, autoencoder_quant_model: DiscAutoEncoder, **kwargs):
        super().__init__(**kwargs)
        self.autoencoder_quant_model = autoencoder_quant_model
        self.quantized_state = []
        self.autoencoder_in = []

    def lstm_layers_forward(self, x, h, c):
        layer_input = x
        for layer_idx in range(self.n_layers):
            ae_in = torch.cat((h[layer_idx], c[layer_idx]), dim=1)
            (h[layer_idx], c[layer_idx]) = torch.chunk(self.autoencoder_quant_model(ae_in), chunks=2, dim=1)
            (h[layer_idx], c[layer_idx]) = self.lstm_layers[layer_idx](layer_input, (h[layer_idx], c[layer_idx]))

            if layer_idx < self.n_layers - 1:
                h[layer_idx] = self.lstm_dropout_layers[layer_idx](h[layer_idx])
            layer_input = h[layer_idx]
