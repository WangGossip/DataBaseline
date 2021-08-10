
# * 各种函数
import csv
from logging import handlers
import time
import os
from matplotlib import lines
import numpy as np
import torch
import matplotlib.pyplot as plt


# * 这部分函数是涉及CSV文件操作（存储训练结果用
def csv_test():
    str_file = 'test.csv'
    f = open(str_file, 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    # 构建表头，实际就是第一行
    csv_writer.writerow(["批次","样本数","正确率"])

    csv_writer.writerow([1,10,0.8])
    csv_writer.writerow([1,20,0.91])
    csv_writer.writerow([1,30,0.85])

    f.close()

# *csv类，记录各种数据
# 文件名：数据
# init:参数args，文件名
# write_title：表头列表
# write_data： 一行内容
# close：关闭
class csv_results:
    def __init__(self, args, str_file='result.csv'):
        csv_path = os.path.join(args.out_path, str_file)
        csv_handler = open(csv_path, 'w', encoding = 'utf-8')
        csv_writer = csv.writer(csv_handler)
        self.csv_path = csv_path
        self.csv_handler = csv_handler
        self.csv_wirter = csv_writer
    
    # 构建表头
    def write_title(self, titles):
        self.csv_wirter.writerow(titles)
        self.count_cols = len(titles)

    # 添加一行内容
    def write_data(self, data):
        if len(data) == self.count_cols:
            self.csv_wirter.writerow(data)
    
    # 关闭表格
    def close(self):
        self.csv_handler.close()

# *一个时间类，作为一个计时器，可以在每次需要计时的时候记录当前用时，也可以返回累计时间、时间总和、平均时间等
class Timer:  #@save
    """记录多次运行时间。"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()

def get_hms_time(sec):
    s_hour = 3600
    s_min = 60
    h = int(sec/s_hour)
    m = int((sec%s_hour)/s_min)
    s = int(sec%s_min)
    return h, m, s
    

# *累加器,每次根据新的数值把当前的数值进行累加(主要用于累计正确率等数据)
class Accumulator:  #@save
    """在`n`个变量上累加。"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# *获取FashionMNIST 标签
# 根据y值获取标签
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# *数据集内容（图片）可视化
# 参数解析
# imgs：存放图像数据，传入的内容要resize为 n*h*w，即 图片数量 x 分辨率
# num_rows, num_cols 代表要展示的布局
# fig_name 是存储的图片名， titles是对应的图片标签
def show_images(imgs, num_rows, num_cols, fig_name, titles=None, scale=2):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    fig.savefig(fig_name)
    return axes

# *绘制图像相关函数
# 画loss（训练）的图
# ~实际用到的参数：全部loss数值；epochs的数量（args）；采样频率
# 需要包括：loss变化趋势（纵坐标），横坐标是epoch变化（需要一个虚线），实际横坐标是epoch.iter, 
# args中应包含保存的路径
# csv格式：[epoch，iter，loss]
def draw_trloss(args, sample_times=5, fig_name='tr_loss.png'):
    # 其他参数
    n_epoch = args.epochs
    save_path = os.path.join(args.out_path, fig_name)
    csv_path = args.csv_record_trloss.csv_path
    # test
    # 手动赋值：save_path、csv_path
    # 获取数据
    f = open(csv_path, 'r')
    csv_reader = csv.reader(f)
    headline = next(csv_reader)
    count_data = 0
    iter_idxs = []
    loss = []
    #~ 读取的str格式，应转换为对应的int或float
    for row in csv_reader:
        count_data += 1             #计数，有多少行 
        iter_idxs.append(int(row[1]))
        loss.append(float(row[2]))
    f.close()
    # -设置横坐标，尺度为1
    # 设置采样相关参数，即每一个迭代epoch中采样几次
    # sample_times = 10
    # epoch，虚线分界；以及采样后的数据
    print('count_data is {}'.format(count_data))
    space_epoch = int(count_data/n_epoch)
    space_sample = int(space_epoch/sample_times)
    print('space_epoch is {}, space_sample is {}'.format(space_epoch, space_sample))
    # epoch_lines = []
    sample_loss_x = []
    sample_loss_y = []

    for i in range(sample_times*n_epoch):
        tmp_x = i*space_sample
        sample_loss_x.append(tmp_x)
        sample_loss_y.append(loss[tmp_x])
    max_loss_y = max(sample_loss_y)
    min_loss_y = min(sample_loss_y)
    # 开始画图
    plt.figure()
    plt.plot(sample_loss_x, sample_loss_y)
    # 添加虚线绘图
    for i in range(1, n_epoch):
        plt.vlines(i*space_epoch, min_loss_y, max_loss_y, colors='g', linestyles='--')
        # epoch_lines.append(i*space_epoch)    
    plt.title("LossResult")
    plt.xlabel("train")
    plt.ylabel("loss")

    plt.savefig(save_path)

# *画acc结果
def draw_tracc(args, fig_name='tr_acc.png'):
    # 参数处理
    n_epoch = args.epochs
    save_path = os.path.join(args.out_path, fig_name)
    csv_path = args.csv_record_tracc.csv_path
    # 读取数据
    f = open(csv_path, 'r')
    csv_reader = csv.reader(f)
    headline = next(csv_reader)
    count_epoch = []
    te_acc = []
    te_loss = []
    for row in csv_reader:
        count_epoch.append(int(row[0]))
        te_loss.append(float(row[1]))
        te_acc.append(float(row[2]))
    f.close()
    # 画图部分
    fig = plt.figure()
    ax_loss = fig.add_subplot(111)
    ax_loss.plot(count_epoch, te_loss, 'r', label='test_loss')
    ax_loss.legend(loc=1)
    ax_loss.set_ylabel('Loss for each epoch')
    ax_acc = ax_loss.twinx() #~重点利用这个函数
    ax_acc.plot(count_epoch, te_acc, 'g', label='test_acc')
    ax_acc.legend(loc=2)
    ax_acc.set_ylabel('Acc for after epoch')
    ax_acc.set_xlabel('Epoch')
    plt.title("Loss&ACC Result")
    plt.savefig(save_path)

# *用于记录的相关函数

# *路径相关函数

# 修改 out-path，这是用来存储最终的log、csv以及对应的图片的
def get_results_dir(args):
    # args.
    time_start = time.localtime()
    path_results_tmp = '{}-{}-{}'.format(args.model_name, args.dataset, time.strftime(".%Y-%m-%d-%H:%M:%S", time_start))
    path_results_fin = os.path.join(args.out_path, path_results_tmp)
    if not os.path.exists(path_results_fin):
        os.makedirs(path_results_fin)
    args.out_path = path_results_fin
    return