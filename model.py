
#*  这个文件用于存放基本的模型

import torch
from torch._C import namedtuple_a_tau
import torch.nn as nn
import torch.nn.functional as F

def get_model(model_name):
    if model_name == 'Net1':
        return Net1

# *这个网络目前还不知道是叫什么结构，现没有名字
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        e1 = F.relu(x)
        x = self.dropout2(e1)
        x = self.fc2(x)
        return x, e1

