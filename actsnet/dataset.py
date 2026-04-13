import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class EEGDataset(Dataset):
    """PyTorch Dataset for loading EEG .npy files.

    Expected directory structure:
        data_dir/
            data.npy    — shape (n_samples, n_channels, n_timesteps)
            labels.npy  — shape (n_samples,) with values 0 or 1

    Or provide arrays directly via `data` and `labels` parameters.
    """

    def __init__(self, data_dir=None, data=None, labels=None, transform=None):
        if data is not None and labels is not None:
            self.data = torch.FloatTensor(data)
            self.labels = torch.LongTensor(labels)
        elif data_dir is not None:
            data_dir = Path(data_dir)
            self.data = torch.FloatTensor(np.load(data_dir / "data.npy"))
            self.labels = torch.LongTensor(np.load(data_dir / "labels.npy"))
        else:
            raise ValueError("Provide either data_dir or (data, labels)")

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class TemporalJitter:
    """Randomly shift the time series by a small offset."""

    def __init__(self, max_shift=5):
        self.max_shift = max_shift

    def __call__(self, x):
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shifts=shift, dims=-1)


class ChannelDropout:
    """Randomly zero out entire channels."""

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, x):
        mask = torch.bernoulli(torch.full((x.shape[0],), 1.0 - self.p))
        return x * mask.unsqueeze(-1)
