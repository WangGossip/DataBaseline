import numpy as np
import torch
import pandas as pd
import gzip
from torch.utils import data


from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


# *获取训练、测试集，根据数据集不同分别处理
def get_dataset(name, data_path):
    # 将PIL转换为tenso
    # trans = transforms.ToTensor()
    if name == 'MNIST' :
        raw_tr = datasets.MNIST(data_path, train=True, download=False)
        raw_te = datasets.MNIST(data_path, train=False, download=False)
        X_tr = raw_tr.data
        Y_tr = raw_tr.targets
        X_te = raw_te.data
        Y_te = raw_te.targets
    elif name == 'FashionMNIST':
        raw_tr = datasets.FashionMNIST(data_path, train=True, download=False)
        raw_te = datasets.FashionMNIST(data_path, train=False, download=False)
        X_tr = raw_tr.data
        Y_tr = raw_tr.targets
        X_te = raw_te.data
        Y_te = raw_te.targets
    elif name == 'CIFAR10':
        raw_tr = datasets.CIFAR10(data_path, train=True, download=False)
        raw_te = datasets.CIFAR10(data_path, train=False, download=False)
        X_tr = raw_tr.data
        Y_tr = torch.tensor(raw_tr.targets)
        X_te = raw_te.data
        Y_te = torch.tensor(raw_te.targets)
    return X_tr, Y_tr, X_te, Y_te


# *重载data类，编写合适的dataloader
def get_handler(name):
    if name == 'MNIST':
        return DataHandler1
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3

class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)