import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch.utils import data
from torch.optim.lr_scheduler import StepLR

# 个人文件函数等
from functions import csv_test, Timer, Accumulator, show_images, get_fashion_mnist_labels
from datasets import get_dataset, get_handler
from model import get_model
import arguments

from d2l import torch as d2l
# *关于这个baseline的函数功能：
# 1. 对于不同数据集开展实验
# 2. 尝试使用不同的网络模型
# 3. 测试一下各种AL采集函数
# 4. 运行结果保存为基线，画图形式即可

#* 单个迭代过程训练函数
def train_epoch(args, model, device, train_loader, optimizer, epoch, Timer):
    model.train()
    for batch_idx, (data, target, idxs) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, e1 = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        # - 记录结果到csv中，并输出到日志（控制台即可）
        if batch_idx % args.log_interval == 0:
            tmp_time = Timer.stop()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, time is {}s'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), tmp_time))
            # -存储到csv中
            if args.dry_run:
                break
# *测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, idxs in test_loader:
            data, target = data.to(device), target.to(device)
            output, e1 = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(args):

    # **测试函数部分
    d2l.load_data_fashion_mnist
    # T = Timer()
    # data_train, labels_train, _, _ = get_dataset('FashionMNIST', './datasets') 
    # X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    # show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
    # total_time = T.stop()
    # ** 测试结束
    # *参数人为赋值
    args.model_name = 'VGG16'
    # 正式的训练过程
    T = Timer()
    # * 参数处理部分
    DATA_NAME = args.dataset
    MODEL_NAME = args.model_name
    # ~todo 不同数据可能要额外计算\
    # -计算一个transform的列表
    transforms_list = {
        'MNIST':
        {
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        },
        'FashionMNIST':
        {
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        },
        'SVHN':
        {
            [transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]
        },
        'CIFAR10':
        {
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
        }
    }
    """
    transform_pool = {'MNIST':
                        {'transform':transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), (0.3081,))])},
                    'FashionMNIST':
                        {'transform':transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), (0.3081,)),transforms.Resize(224)])},
                    'SVHN':
                        {'transform':transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])},
                    'CIFAR10':
                        {'transform':transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])}                                                        
    }
    """
    tmp_transform_list = transforms_list[DATA_NAME]
    if MODEL_NAME[:3] == 'VGG':
        tmp_transform_list.append(transforms.Resize(224))
    # 是否使用GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # *加载数据集，设置dataloader等
    # 数据集相关参数
    transform = transforms.Compose(tmp_transform_list)
    
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    tmp_t = T.stop()
    print('Deal args time is {} s'.format(tmp_t))
    # 读取数据
    X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, args.data_path)
    handler = get_handler(DATA_NAME)

    # -可能有额外的数据处理部分（初始数据集）
    train_loader = DataLoader(handler(X_tr, Y_tr, transform=transform), **train_kwargs)
    test_loader = DataLoader(handler(X_te, Y_te, transform=transform), **test_kwargs)

    tmp_t = T.stop()
    print('Read Data time is {} s'.format(tmp_t))

    # *模型训练部分
    # 加载、选择模型，设置优化器、处理相关参数
    net = get_model(MODEL_NAME).to(device)
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    n_epoch = args.epochs

    for epoch in range(1, n_epoch+1):
        train_epoch(args, net, device, train_loader, optimizer, n_epoch, T)
        test(net, device, test_loader)
        scheduler.step()
    tmp_t = T.stop()
    print('Total Data time is {} s'.format(tmp_t))
if __name__ == '__main__' :
    args = arguments.get_args()
    main(args)