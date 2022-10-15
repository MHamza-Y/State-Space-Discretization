import dill
import numpy as np
import torch

from state_quantization.quantization_models import ForcastingDiscFinalState


class NormalizeTransform:
    def __init__(self, save_path='NormalizeDatasetInstanceConfigs'):
        self.save_path = save_path
        self.mean = 0
        self.std = 1

    def compute_mean_std(self, x):
        self.mean = x.mean(dim=(0, 1), keepdim=True)
        self.std = x.std(dim=(0, 1), keepdim=True)

    def transform(self, x):
        x = (x - self.mean) / self.std

        return x

    def inverse_transform(self, x):
        x = (x * self.std) + self.mean
        return x

    def save(self):
        with open(self.save_path, 'wb') as f:
            dill.dump(self, f)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    @classmethod
    def load(cls, save_path):
        with open(save_path, 'rb') as f:
            return dill.load(f)


class Bin2Dec:

    def __call__(self, binary_array):
        bits = binary_array.size()[-1]
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(binary_array.device)
        return torch.sum(mask * binary_array, -1)


class LSTMQuantize:

    def __init__(self, model: ForcastingDiscFinalState, normalize_transformer, reshape):
        self.model = model

        self.reshape = reshape
        self.bin2dec = Bin2Dec()
        self.device = model.get_device()
        self.normalize_transformer = normalize_transformer

    def __call__(self, x):
        x = np.array(x).astype(np.float32)

        x = torch.from_numpy(x).to(self.device)
        x = x.view(self.reshape)
        x = torch.flip(x, [1])
        x = self.normalize_transformer.transform(x)
        x = torch.nan_to_num(x, 1)
        self.model(x)
        return self.bin2dec(self.model.quantized_state).tolist()


class QuantizeBuffer:
    def __init__(self, lstm_quantize, keys):
        self.keys = keys
        self.lstm_quantize = lstm_quantize

    def __call__(self, buffer):
        for key in self.keys:
            buffer[key] = self.lstm_quantize(buffer[key])
        return buffer


def quantize_transform_creator(device, keys, reshape=(-1, -1, 6)):
    model_path = 'state_quantization/model'

    model = torch.load(model_path).to(device)
    model.eval()
    model.set_look_ahead(0)
    normalize_dataset = NormalizeTransform.load('state_quantization/NormalizeInputConfigs.pkl')
    normalize_dataset.to(device)
    lstm_quantize = LSTMQuantize(model=model, normalize_transformer=normalize_dataset, reshape=reshape)
    return QuantizeBuffer(lstm_quantize=lstm_quantize, keys=keys)
