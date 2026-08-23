# -- coding: utf-8 --
import numpy as np
import torch
from torch.utils.data import Dataset

class TerminalDataset(Dataset):
    """码头作业量时序数据集（改进维度处理）"""
    def __init__(self, data, seq_length=6):
        self.original_shape = data.shape
        self.data = torch.FloatTensor(data.flatten())
        self.seq_length = seq_length
    def __len__(self):
        return len(self.data) - self.seq_length - 1
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length].view(-1, 1)
        y = self.data[idx + self.seq_length]
        return x, y
    def restore_shape(self, data):
        return data.reshape(self.original_shape)

class DataProcessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def normalize(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return data.reshape(-1, 1) * self.std + self.mean

