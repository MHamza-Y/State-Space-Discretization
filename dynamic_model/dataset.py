import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class DynamicsModelDataset(Dataset):
    def __init__(self, x, y, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
        self.len = x.shape[0]

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

    def get_features_size(self):
        return self.x.shape[2]

    def get_seq_len(self):
        return self.x.shape[1]

    def get_output_feature_size(self):
        return self.y.shape[2]
