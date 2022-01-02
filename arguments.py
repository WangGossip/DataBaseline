import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch DataBaseline Framework')
    parser.add_argument('--no-cuda', action='store_true', 
                        help='If training is to be done on a GPU')
    # *数据集相关参数
    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='D', 
                        help='Name of the dataset used.')
    parser.add_argument('--data-path', type=str, default='./datasets', 
                        help='Path to where the data is')                      
    # *训练&测试相关参数 
    parser.add_argument('--model-name', type=str, default='Net3', 
                        help='Model used for training (default:Net1)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', 
                        help='Batch size used for training (defaule:128)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N', 
                        help='Batch size used for training (defaule:1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='Number of epochs for training (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='Learning rate (default: 0.1)')         
    parser.add_argument('--no-sch', action='store_true', 
                        help='If to use a scheduler')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)') 
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Learning momentum (default: 0.9)') 
    parser.add_argument('--opt', type=str, default='sgd', metavar='S',
                        help='Optimizer used for training (default:sgd)')
    parser.add_argument('--sch', type=str, default='cos', metavar='S', 
                        help='Scheduler used for training and optimizer(default:CosineAnnealingLR)')
    parser.add_argument('--tmax', type=int, default=80, metavar='N',
                        help='T_max in cos scheduler'    )
    parser.add_argument('--step-size', type=int, default=5, 
                        help='Learning rate step size (default: 5)') 
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Quickly check a single pass')       
    # *日志、模型、结果等存储相关参数
    parser.add_argument('--logs-path', type=str, default='./logs',
                        help='Path to save logs')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-name', type=str, default='run_record', 
                        help='Final performance of the models will be saved with this name')
    parser.add_argument('--log-level', type=str, default='debug', 
                        help='Level for writing log')
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
