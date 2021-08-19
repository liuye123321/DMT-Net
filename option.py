import argparse
import utility
parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=2,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--labeled_bs', type=int, default=8, help='labeled_batch_size per gpu')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--dataroot', type=str, default='/media/ext2/liuye/dataset/our_new/train',
                    help='dataset directory')
parser.add_argument('--unlb_dataroot', type=str, default='/media/ext2/liuye/dataset/domain/train/unlabeled',
                    help='unlabeled dataset directory')
parser.add_argument('--unlb_depthroot', type=str, default='/media/ext2/liuye/dataset/domain/train/unlabeled_depth',
                    help='unlabeled dataset directory')
parser.add_argument('--testpath', type=str, default='/media/ext2/liuye/dataset/our_new/test',
                    help='dataset directory for testing')

parser.add_argument('--dataset', type=str, default='our', #our
                    help='ablation study dataset')

parser.add_argument('--data_train', type=str, default='new_data_train',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='new_data_val',
                    help='test dataset name')
parser.add_argument('--fineSize', type=int, default=128, help='then crop to this size')
# parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--no_flip', type=bool, default=False,help='if specified, do not flip the images for data augmentation')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--nAveGrad', type=int, default=16)  #16
parser.add_argument('--n_phase', type=str, default='train',
                    help='train or test')
parser.add_argument('--test_type', type=str, default='syn',
                    help='syn or real')
parser.add_argument('--flip', type=int, default=1)
# Model specifications
parser.add_argument('--model', default='DID',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dual_weight', type=float, default=0.1,
                    help='the weight of dual loss')
parser.add_argument('--dual_flag', type=bool, default=False,
                    help='dual or not')
# Training specifications
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=200,  #(epochs = max_iters/训练集数目*batch_size)
                    help='number of epochs to train')
parser.add_argument('--warm_up_epoch', type=int, default=5,
                    help='number of warm_up_epoch')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=10,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    choices=('step', 'exponent', 'lambda'),
                    help='learning rate decay type')
parser.add_argument('--power', type=float, default=0.9,
                    help='lambda')
parser.add_argument('--gamma', type=float, default=0.85,
                    help='learning rate decay factor for step decay')

parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='DID',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--print_every', type=int, default=200,
                    help='how many batches to wait before logging training status')



## drn config
parser.add_argument('--eta_min', type=float, default=1e-7,
                    help='eta_min lr')
parser.add_argument('--negval', type=float, default=0.2,
                    help='Negative value parameter for Leaky ReLU')
parser.add_argument('--pre_train_dual', type=str, default='.',
                    help='pre-trained dual model directory')


parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')

args = parser.parse_args()
utility.init_model(args)

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

