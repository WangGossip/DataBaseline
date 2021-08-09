from logging.handlers import HTTPHandler
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
from functions import csv_test, Timer, Accumulator, show_images, get_fashion_mnist_labels, get_results_dir, csv_results, get_hms_time, draw_trloss, draw_tracc
from datasets import get_dataset, get_handler
from model import get_model
from log import Logger
import arguments

# from d2l import torch as d2l
# *关于这个baseline的函数功能：
# 1. 对于不同数据集开展实验
# 2. 尝试使用不同的网络模型
# 3. 测试一下各种AL采集函数
# 4. 运行结果保存为基线，画图形式即可

#* 单个迭代过程训练函数
def train_epoch(args, model, device, train_loader, optimizer, epoch):
    # 部分参数处理
    T = args.timer
    log_run = args.log_run
    csv_record_trloss = args.csv_record_trloss

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
            tmp_time = T.stop()
            log_run.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, time is {:.4f} s'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), tmp_time))
            csv_record_trloss.write_data([epoch, batch_idx, loss.item()])
            T.start()

            if args.dry_run:
                break

# *测试函数
def test(args, model, device, test_loader, epoch):
    # 部分参数处理
    T = args.timer
    log_run = args.log_run
    csv_record_tracc = args.csv_record_tracc

    len_testdata = len(test_loader.dataset)
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

    test_loss /= len_testdata
    acc = correct / len_testdata

    tmp_time = T.stop()
    log_run.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), time is {:.4f} s'.format(
        test_loss, correct, len_testdata,
        100. * correct / len_testdata, tmp_time))
    csv_record_tracc.write_data([epoch, test_loss, acc])
    T.start()

def test_draw(args):
    str_path = './results/VGG16-FashionMNIST-.2021-08-07-22:46:54/train_loss.csv'
    draw_trloss(args, str_path)

def main(args):
    #test 测试函数部分
    # T = Timer()
    # data_train, labels_train, _, _ = get_dataset('FashionMNIST', './datasets') 
    # X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    # show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
    # total_time = T.stop()
    #test 测试结束
    # 参数人为赋值
    # args.model_name = 'VGG16'
    # 正式的训练过程
    # * 参数处理部分
    # args内添加Timer类，计时用
    T = Timer()
    args.timer = T
    # 处理存储文件夹，args.out_path代表结果输出位置
    get_results_dir(args)
    # args内添加csv类
    csv_record_trloss = csv_results(args, 'train_loss.csv')
    csv_record_tracc = csv_results(args, 'train_acc.csv')
    args.csv_record_trloss = csv_record_trloss
    args.csv_record_tracc = csv_record_tracc
    # logger类
    log_run = Logger(args, level=args.log_level)
    args.log_run = log_run
    # 部分会常用的变量
    DATA_NAME = args.dataset
    MODEL_NAME = args.model_name

    tmp_t = T.stop()
    log_run.logger.debug('程序开始，部分基础参数处理完成，用时 {:.4f} s'.format(tmp_t))
    log_run.logger.debug('使用数据集为：{}， 网络模型为：{}， epoch为：{}， batchsize为：{}'.
                        format(DATA_NAME, MODEL_NAME, args.epochs, args.batch_size))
    T.start()

    # test bug:2个results
    # csv_record_trloss.close()
    # csv_record_tracc.close()
    # return

    # -计算一个transform的列表
    transforms_list = {
        'MNIST':
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
        'FashionMNIST':
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
        'SVHN':
            [transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))],
        'CIFAR10':
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
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
    #- VGG网络需要resize为224
    if MODEL_NAME[:3] == 'VGG':
        tmp_transform_list.append(transforms.Resize(224))
    # 是否使用GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # *加载数据集，设置dataloader等
    # 数据集相关参数，包括transform以及batchsize等合并参数
    transform = transforms.Compose(tmp_transform_list)
    
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    tmp_t = T.stop()
    log_run.logger.debug('处理transfom、cuda等参数，用时 {:.4f} s'.format(tmp_t))
    T.start()

    # *读取数据
    X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, args.data_path)
    handler = get_handler(DATA_NAME)

    # -可能有额外的数据处理部分（初始数据集）
    train_loader = DataLoader(handler(X_tr, Y_tr, transform=transform), **train_kwargs)
    test_loader = DataLoader(handler(X_te, Y_te, transform=transform), **test_kwargs)

    # test开始测试
    # X, y, _ = next(iter(train_loader))
    # show_images(X.reshape(64, 224, 224), 8, 8, 'VGG_show.png',titles=get_fashion_mnist_labels(y));
    # return
    # test测试结束
    # ~是否使用resize没有影响

    tmp_t = T.stop()
    log_run.logger.debug('读取数据用时 {:.4f} s'.format(tmp_t))
    T.start()

    time_start_train = time.time()
    # *模型训练部分
    # 加载、选择模型，设置优化器、处理相关参数
    net = get_model(MODEL_NAME).to(device)
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    n_epoch = args.epochs

    csv_record_trloss.write_title(['epoch', 'batch', 'loss'])
    csv_record_tracc.write_title(['epoch', 'loss', 'acc'])
    
    for epoch in range(1, n_epoch+1):
        train_epoch(args, net, device, train_loader, optimizer, epoch)
        test(args, net, device, test_loader, epoch)
        scheduler.step()
    
    time_train = time.time() - time_start_train
    h_tmp, m_tmp, s_tmp = get_hms_time(time_train)
    log_run.logger.info('训练用时：{} h {} min {} s'.format(h_tmp, m_tmp, s_tmp))

    # test 暂时不用了吧 看csv也是一样的
    # 运行结束，各种关闭函数
    csv_record_trloss.close()
    csv_record_tracc.close()
    # 画图
    T.start()
    draw_trloss(args)
    draw_tracc(args)
    tmp_t=T.stop()
    log_run.logger.info('画图用时：{:.4f} s'.format(tmp_t))
    log_run.logger.info('程序结束，\n运行log存储路径为：{}\n实验结果存储路径为：{}'.format(args.log_run.filename,args.out_path))

if __name__ == '__main__' :
    args = arguments.get_args()
    # test_draw(args)
    main(args)