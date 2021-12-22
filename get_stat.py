import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

def getStat(x_tr, y_tr):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    len_data = len(x_tr)
    print(len_data)
    # train_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=1, shuffle=False, num_workers=0,
    #     pin_memory=True)
    train_kwargs = {'batch_size': 1, 'shuffle':False}
    train_loader = DataLoader(DataHandler3(x_tr, y_tr, transforms.Compose([transforms.ToTensor()])), **train_kwargs)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for batch_idx, (data, target, idxs) in enumerate(train_loader):
    # for X in train_loader:
        # print(X)
        # break
        data = data.float()
        for d in range(3):
            mean[d] += data[:, d, :, :].mean()
            std[d] += data[:, d, :, :].std()
    mean.div_(len_data)
    std.div_(len_data)
    return list(mean.numpy()), list(std.numpy())

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

if __name__ == '__main__':
    raw_tr = datasets.CIFAR10('./datasets', train=True, download=False)
    x_tr = raw_tr.data
    y_tr = torch.tensor(raw_tr.targets)
    print(getStat(x_tr, y_tr))