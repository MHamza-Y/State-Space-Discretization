import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class DynamicsModelDataset(Dataset):
    def __init__(self, x, y, device, normalize=False, replace_constant_val=0.7):
        self.x = x
        self.y = y
        self.len = x.shape[0]
        if normalize:
            self.x_mean = self.x.mean(dim=(0, 1), keepdim=True)
            self.y_mean = self.y.mean(dim=(0, 1), keepdim=True)
            self.x_std = self.x.std(dim=(0, 1), keepdim=True)
            self.y_std = self.y.std(dim=(0, 1), keepdim=True)
            self.x = (self.x - self.x_mean) / self.x_std
            self.x[:, :, 0] = replace_constant_val

            self.y = (self.y - self.y_mean) / self.y_std
            self.y[:, :, 0] = replace_constant_val
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


def load_dataset(file_path, input_key, output_key, dataset_class, device, normalize=False, test_size=0.3, ):
    dataset = np.load(file_path, allow_pickle=True)
    x = dataset[()][input_key]
    y = dataset[()][output_key]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_dataset = dataset_class(x_train, y_train, device, normalize=normalize)
    validation_dataset = dataset_class(x_test, y_test, device, normalize=normalize)
    return train_dataset, validation_dataset
