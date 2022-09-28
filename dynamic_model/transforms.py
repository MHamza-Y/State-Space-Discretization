import dill


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
