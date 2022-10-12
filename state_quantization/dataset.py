import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from state_quantization.transforms import NormalizeTransform


class DynamicsModelDataset(Dataset):
    def __init__(self, x, y, normalized=False):
        self.x = x
        self.y = y
        self.len = x.shape[0]

        self.normalized = normalized
        print(x.shape)
        print(y.shape)

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

    def get_features_size(self):
        return self.x.shape[2]

    def get_seq_len(self):
        return self.x.shape[1] - self.get_look_ahead_size()

    def get_output_feature_size(self):
        return self.y.shape[2]

    def get_output_seq_len(self):
        return self.y.shape[1]

    def get_look_ahead_size(self):
        return self.get_output_seq_len() - 1


def load_dataset(file_path, input_key, output_key, dataset_class, device, normalize=False, test_size=0.3,
                 y_clip_range=None, normalized_data_params_save_path='NormalizeInputConfigs'):
    dataset = np.load(file_path, allow_pickle=True)
    x = dataset[()][input_key]
    y = dataset[()][output_key]
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    if y_clip_range:
        print('Clipping y')
        y = y[:, :, y_clip_range[0]:y_clip_range[1]]

    normalize_dataset_x = NormalizeTransform(save_path=normalized_data_params_save_path)
    normalize_dataset_y = NormalizeTransform()
    if normalize:
        normalize_dataset_x.compute_mean_std(x)
        normalize_dataset_y.compute_mean_std(y)
        x = normalize_dataset_x.transform(x)
        y = normalize_dataset_y.transform(y)
        normalize_dataset_x.save()
    x[:, :, 0] = 1
    if not y_clip_range:
        y[:, :, 0] = 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    train_dataset = dataset_class(x_train, y_train, normalized=normalize)
    validation_dataset = dataset_class(x_test, y_test, normalized=normalize)
    return train_dataset, validation_dataset
