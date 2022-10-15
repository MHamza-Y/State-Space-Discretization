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
