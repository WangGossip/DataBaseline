
#*  这个文件用于存放基本的模型

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def get_model(model_name):
    out_channels=[64,128,256,512,512]
    if model_name == 'Net1':
        return Net1()
    elif model_name == 'VGG16':
        return VGG([2,2,3,3,3], num_classes=10)
    elif model_name =='VGG11':
        return VGG([1,1,2,2,2], num_classes=10)
    # -这是一个小型的，通道数都减少为1/4
    elif model_name =='VGG11s':
        return VGG([1,1,1,2,2], out_channels=[i//4 for i in out_channels])
        # conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        # ratio = 4
        # small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
        # return vgg11(small_conv_arch)



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

class VGG(nn.Module):
    """
    VGG builder
    """
    def __init__(self, arch: object, num_classes=10, out_channels=[64,128,256,512,512]) -> object:
        super(VGG, self).__init__()
        # - 输入的通道数（用于两个MNIST数据集）
        self.in_channels = 1
        self.conv3_64 = self.__make_layer(out_channels[0], arch[0])
        self.conv3_128 = self.__make_layer(out_channels[1], arch[1])
        self.conv3_256 = self.__make_layer(out_channels[2], arch[2])
        self.conv3_512a = self.__make_layer(out_channels[3], arch[3])
        self.conv3_512b = self.__make_layer(out_channels[4], arch[4])
        
        self.fc1 = nn.Linear(7*7*out_channels[4], 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        e1 = out
        out = self.fc3(out)
        out = F.softmax(out)
        return out, e1

# ~下面是d2l中实现的VGG
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)    
def vgg11(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks, nn.Flatten(),
                         # 全连接层部分
                         nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
                         nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
                         nn.Dropout(0.5), nn.Linear(4096, 10))