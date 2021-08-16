import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NumpyDataset(Dataset):    
    def __init__(self, data_path, labels_path, transform, three_channels=False):
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')

        self.transform = transform
        self.three_channels = three_channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.three_channels:
            data = np.tile(data, (3, 1, 1)) # copies 2d array to the other 3 channels
        
        data = data.astype(np.float32)
        data = self.transform(torch.from_numpy(data))

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return data, label

def get_dataloader(data_path, labels_path, augmentations, bs, three_channels=False, num_workers=2, shuffle=False):
    # create dataset
    dataset = NumpyDataset(data_path, labels_path, augmentations, three_channels)
    
    # source dataloader
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=bs, num_workers=num_workers)
    
    return dataloader