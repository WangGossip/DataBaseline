import numpy as np
import time
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch.utils import data

# 个人文件函数等
from functions import csv_test, Timer, Accumulator, show_images, get_fashion_mnist_labels
from datasets import get_dataset, get_handler
import arguments

# *关于这个baseline的函数功能：
# 1. 对于不同数据集开展实验
# 2. 尝试使用不同的网络模型
# 3. 测试一下各种AL采集函数
# 4. 运行结果保存为基线，画图形式即可
def main(args):
    # **测试函数部分
    # T = Timer()
    # data_train, labels_train, _, _ = get_dataset('FashionMNIST', './datasets') 
    # X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    # show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
    # total_time = T.stop()
    # ** 测试结束

    # 正式的训练过程
    T = Timer()
    # * 参数处理部分
    DATA_NAME = args.dataset
    # todo 不同数据可能要额外计算
    transform_pool = {'MNIST':
                        {'transform':transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), (0.3081,))])},
                    'FashionMNIST':
                        {'transform':transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), (0.3081,))])},
                    'SVHN':
                        {'transform':transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])},
                    'CIFAR10':
                        {'transform':transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])}                                                        
    }
    # 是否使用GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    # *加载数据集，设置dataloader等
    # 数据集相关参数
    transform = transform_pool[DATA_NAME]['transform']
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
    # 加载、选择模型


    tmp_t = T.stop()
    print('Draw Data time is {} s'.format(tmp_t))
if __name__ == '__main__' :
    args = arguments.get_args()
    main(args)