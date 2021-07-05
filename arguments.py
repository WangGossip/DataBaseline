import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch DAL Framework')
    parser.add_argument('--no-cuda', action='store_true', 
                        help='If training is to be done on a GPU')
    # *数据集相关参数
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='D', 
                        help='Name of the dataset used.')
    parser.add_argument('--data-path', type=str, default='./../../datasets', 
                        help='Path to where the data is')
    # *AL采样相关参数
    parser.add_argument('--method', type=str, default='RS', 
                        help='strategy used for sampling')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--prop-init', type=float, default='0.2', metavar='R', 
                        help='Proportion of sample initialization')
    parser.add_argument('--prop-budget', type=float, default='0.4', metavar='R', 
                        help='Total proportion of sample budget')
    parser.add_argument('--times', type=int, default=10, metavar='T', 
                        help='Times of sampling')                        
    # *训练&测试相关参数 
    parser.add_argument('--model-name', type=str, default='Net1', 
                        help='Model used for training (default:Net1)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', 
                        help='Batch size used for training (defaule:64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', 
                        help='Batch size used for training (defaule:1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='Number of epochs for training (default: 20)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='Learning rate (default: 1.0)')         
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)') 
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Quickly check a single pass')       
    # *日志、模型存储相关参数
    parser.add_argument('--logs-path', type=str, default='./logs',
                        help='Path to save logs')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-name', type=str, default='accuracies.log', 
                        help='Final performance of the models will be saved with this name')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For saving the current Model')
    parser.add_argument('--save-results', action='store_false', default=True, 
                        help='For saving the train results')
    parser.add_argument('--out-path', type=str, default='./results', 
                        help='Path to where the output log will be')

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    if not os.path.exists(args.logs_path):
        os.mkdir(args.logs_path)
    return args