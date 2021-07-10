
# * 各种函数
import csv
import time
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

# *画图相关
# *获取FashionMNIST 标签
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# *数据集内容（图片）可视化
def show_images(imgs, num_rows, num_cols, titles=None, scale=2):  #@save
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
    fig.savefig('test.png')
    return axes

# *绘制动画