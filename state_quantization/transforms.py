import os

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
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'wb') as f:
            dill.dump(self, f)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

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
        self.model.eval()

        self.reshape = reshape
        self.bin2dec = Bin2Dec()
        self.device = model.get_device()
        self.normalize_transformer = normalize_transformer
        self.y = None

    def get_continuous_output(self):
        return self.y

    def __call__(self, x):
        x = np.array(x).astype(np.float32)

        x = torch.from_numpy(x).to(self.device)
        x = x.view(self.reshape)
        x = torch.flip(x, [1])
        x = self.normalize_transformer.transform(x)
        x = torch.nan_to_num(x, 1)
        self.y = self.model(x)
        return self.bin2dec(self.model.quantized_state).tolist()


class QuantizeBuffer:
    def __init__(self, lstm_quantize, keys):
        self.keys = keys
        self.lstm_quantize = lstm_quantize

    def __call__(self, buffer):
        for key in self.keys:
            buffer[key] = self.lstm_quantize(buffer[key])
        return buffer


class MultiModelQuantizeBuffer:

    def __init__(self, lstm_quantize_transforms, keys, tags):
        self.lstm_quantize_transforms = lstm_quantize_transforms
        self.keys = keys
        self.tags = tags

    def __call__(self, buffer):
        for key in self.keys:
            for i, lstm_quantize in enumerate(self.lstm_quantize_transforms):
                buffer[f"{self.tags[i]}_{key}"] = lstm_quantize(buffer[key])
            buffer.pop(key, None)

        return buffer


def load_lstm_quantizer(model_path, device, reshape):
    model = torch.load(model_path).to(device)
    model.eval()
    model.set_look_ahead(0)
    normalize_dataset = NormalizeTransform.load('tmp/transformer/NormalizeInputConfigs.pkl')
    normalize_dataset.to(device)
    return LSTMQuantize(model=model, normalize_transformer=normalize_dataset, reshape=reshape)


def multi_model_quantize_transforms_creator(model_paths, device, keys, reshape=(-1, -1, 6)):
    lstm_quantize_transforms = []
    tags = []
    for model_path in model_paths:
        lstm_quantize = load_lstm_quantizer(model_path=model_path, device=device, reshape=reshape)
        lstm_quantize_transforms.append(lstm_quantize)
        tag = model_path.split("/")[-1]
        tags.append(tag)
    return MultiModelQuantizeBuffer(lstm_quantize_transforms=lstm_quantize_transforms, keys=keys, tags=tags)


def quantize_transform_creator(model_path, device, keys, reshape=(-1, -1, 6)):
    lstm_quantize = load_lstm_quantizer(model_path=model_path, device=device, reshape=reshape)
    return QuantizeBuffer(lstm_quantize=lstm_quantize, keys=keys)
