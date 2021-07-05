
# * 各种函数
import csv

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

