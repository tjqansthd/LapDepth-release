import argparse

parser = argparse.ArgumentParser(description='Laplacian Depth Residual Network training on KITTI',
									formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory setting
parser.add_argument('--model_dir',type=str, default = '')
parser.add_argument('--trainfile_kitti', type=str, default = "./datasets/eigen_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_kitti', type=str, default = "./datasets/eigen_test_files_with_gt_dense.txt")
parser.add_argument('--trainfile_nyu', type=str, default = "./datasets/nyudepthv2_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_nyu', type=str, default = "./datasets/nyudepthv2_test_files_with_gt_dense.txt")
parser.add_argument('--data_path', type=str, default = "./datasets/KITTI")
parser.add_argument('--use_dense_depth', action='store_true', help='using dense depth data for gradient loss')

# Optimizer and dataloader setting
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N',help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--batch_size', default=16, type=int, metavar='N', help='mini-batch size')

parser.add_argument('--lr', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--end_lr', default=0.00001, type=float, metavar='LR', help='ending learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight_decay', '--wd', default=1e-2, type=float, metavar='W', help='weight decay')
parser.add_argument('--adam_eps', type=float, help='epsilon in Adam optimizer', default=1e-3)
parser.add_argument('--dataset', type=str, default = "KITTI")

# Logging setting
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--val_in_train', action='store_true', help='validation process in training')

# Training and Testing setting
parser.add_argument('--encoder', type=str, default = "ResNext101")
parser.add_argument('--norm', type=str, default = "BN")
parser.add_argument('--act', type=str, default = "ReLU")
parser.add_argument('--img_save', action='store_true', help='result image save')
parser.add_argument('--cap', default=80.0, type=float, metavar='MaxVal', help='cap setting for kitti eval')
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')
parser.add_argument('--height', type=int, default = 352)
parser.add_argument('--width', type=int, default = 704)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--lv6', action='store_true', help='use lv6 Laplacian decoder')

# GPU parallel process setting
parser.add_argument('--gpu_num', type=str, default = "0,1,2,3", help='force available gpu index')
parser.add_argument('--distributed', action='store_true')
parser.add_argument("--local_rank", type=int)
parser.add_argument('--gpu', type=int, default = 0)
parser.add_argument('--world_size', type=int, default = 1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)

args = parser.parse_args()