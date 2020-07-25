import os
import torch
from torch.utils.data import DataLoader,TensorDataset

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, scans, labels):
        'Initialization'
        self.labels = labels
        self.scans = scans

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.scans)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.scans[index]

        # Load tf record data and convert to torch tensor
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y