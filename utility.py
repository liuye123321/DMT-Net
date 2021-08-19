import os
import math
import time
import datetime
from functools import reduce

# import matplotlib
#matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc
import cv2
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
from PIL import Image
def tensor2im(image_tensor, imtype=np.uint8):
	image_numpy = image_tensor[0].cpu().float().numpy()
	# image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
	image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
	return image_numpy.astype(imtype)

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
	return window

def SSIM1(img1, img2):
	(_, channel, _, _) = img1.size()
	window_size = 11
	pad = int(window_size/11)
	window = create_window(window_size, channel).to(img1.device)
	mu1 = F.conv2d(img1, window, padding = pad, groups = channel)
	mu2 = F.conv2d(img2, window, padding = pad, groups = channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1*mu2

	sigma1_sq = F.conv2d(img1*img1, window, padding = pad, groups = channel) - mu1_sq
	sigma2_sq = F.conv2d(img2*img2, window, padding = pad, groups = channel) - mu2_sq
	sigma12 = F.conv2d(img1*img2, window, padding = pad, groups = channel) - mu1_mu2

	C1 = 0.01**2
	C2 = 0.03**2

	ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
	return ssim_map.mean()


def SSIM(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
	# Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
	if val_range is None:
		if torch.max(img1) > 128:
			max_val = 255
		else:
			max_val = 1

		if torch.min(img1) < -0.5:
			min_val = -1
		else:
			min_val = 0
		L = max_val - min_val
	else:
		L = val_range

	padd = 0
	(_, channel, height, width) = img1.size()
	if window is None:
		real_size = min(window_size, height, width)
		window = create_window(real_size, channel=channel).to(img1.device)

	mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
	mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1 * mu2

	sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
	sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
	sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

	C1 = (0.01 * L) ** 2
	C2 = (0.03 * L) ** 2

	v1 = 2.0 * sigma12 + C2
	v2 = sigma1_sq + sigma2_sq + C2
	cs = torch.mean(v1 / v2)  # contrast sensitivity

	ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

	if size_average:
		ret = ssim_map.mean()
	else:
		ret = ssim_map.mean(1).mean(1).mean(1)

	if full:
		return ret, cs
	return ret



def PSNR(img1, img2):
	img1 = img1[0].cpu().float().numpy()
	img2 = img2[0].cpu().float().numpy()
	mse = np.mean((img1 - img2) ** 2)
	if mse == 0:
		return 100
	PIXEL_MAX = 255
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            # self.dir = '../experiment/' + args.save
            self.dir = '/media/ext2/liuye/experiment'+args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))


        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results/')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch):
        trainer.model.save(self.dir, epoch)
        trainer.loss.save(self.dir)
        # trainer.loss.plot_loss(self.dir, epoch)

        # self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()


    def save_image(self, image_numpy, image_path):
        image_pil = None
        if image_numpy.shape[2] == 1:
            image_numpy = np.reshape(image_numpy, (image_numpy.shape[0], image_numpy.shape[1]))
            image_pil = Image.fromarray(image_numpy, 'L')
        else:
            image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_path)

    def save_images(self, image_name, visuals, type):
        filename = '{}/results/'.format(self.dir)
        # image_name = '%s_%s.png' % (image_name, type)
        image_name = '%s.png' % (image_name)
        save_path = os.path.join(filename, image_name)
        self.save_image(visuals, save_path)


    # def save_results_misc(self, filename, save_list):
    #     filename = '{}/results/misc/{}'.format(self.dir, filename)
    #     postfix = ('dehaze', 'haze', 'gt')
    #     for v, p in zip(save_list, postfix):
    #         normalized = v[0].data.mul(255 / self.args.rgb_range)
    #         ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
    #         misc.imsave('{}{}.png'.format(filename, p), ndarr)
    #
    # def save_results_vutils(self, filename, save_list):
    #     filename = '{}/results/vutils/{}'.format(self.dir, filename)
    #     postfix = ('dehaze', 'haze', 'gt')
    #     for v, p in zip(save_list, postfix):
    #         vutils.save_image(v[0].data, '{}{}.png'.format(filename, p),
    #                           normalize=True, scale_each=False)

def quantize(img):
    return img.clamp(0, 1)

def make_optimizer(args, my_model):
    # if args.model == 'DRN_v1':
    #     if args.n_GPUs>1:
    #         ignore_params1 = list(map(id, my_model.model.module.layer0.parameters()))
    #         ignore_params2 = list(map(id, my_model.model.module.layer1.parameters()))
    #         ignore_params3 = list(map(id, my_model.model.module.layer2.parameters()))
    #     else:
    #         ignore_params1 = list(map(id, my_model.model.layer0.parameters()))
    #         ignore_params2 = list(map(id, my_model.model.layer1.parameters()))
    #         ignore_params3 = list(map(id, my_model.model.layer2.parameters()))
    #     trainable = filter(lambda p: id(p) not in ignore_params1 + ignore_params2 + ignore_params3,
    #                        my_model.parameters())
    # else:
    #     trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)


# def make_dual_optimizer(opt, dual_models):
#     dual_optimizers = []
#
#     for dual_model in dual_models:
#         for para in dual_model.parameters():
#             para.requires_grad = False
#         temp_dual_optim = torch.optim.Adam(
#             params=dual_model.parameters(),
#             lr=opt.lr,
#             betas=(opt.beta1, opt.beta2),
#             eps=opt.epsilon,
#             weight_decay=opt.weight_decay)
#         dual_optimizers.append(temp_dual_optim)
#     return dual_optimizers
def make_scheduler(args, my_optimizer):
    if args.decay_type == 'lambda':
        def lambda_rule(epoch):
            epoch = epoch + 1
            lr_l = (1 - (iter / args.max_iter))**args.power

            # lr_l = epoch / args.warm_up_epoch if epoch <= args.warm_up_epoch else 0.5 * \
            #             (math.cos((epoch - args.warm_up_epoch) / (args.epochs - args.warm_up_epoch) * math.pi) + 1)
            return lr_l
        scheduler = lrs.LambdaLR(my_optimizer, lr_lambda=lambda_rule)
    elif args.decay_type == 'step':
        scheduler = lrs.StepLR(my_optimizer, step_size=args.lr_decay, gamma=args.gamma)
    elif args.decay_type == 'exponent':
        scheduler = lrs.ExponentialLR(my_optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler
# def make_scheduler(args, my_optimizer):
#     if args.decay_type == 'step':
#         scheduler = lrs.StepLR(
#             my_optimizer,
#             step_size=args.lr_decay,
#             gamma=args.gamma
#         )
#     elif args.decay_type.find('step') >= 0:
#         milestones = args.decay_type.split('_')
#         milestones.pop(0)
#         milestones = list(map(lambda x: int(x), milestones))
#         scheduler = lrs.MultiStepLR(
#             my_optimizer,
#             milestones=milestones,
#             gamma=args.gamma
#         )
#     return scheduler
# def make_scheduler(opt, my_optimizer):
#     scheduler = lrs.CosineAnnealingLR(
#         my_optimizer,
#         float(opt.epochs),
#         eta_min=opt.eta_min
#     )
#
#     return scheduler
# def make_dual_scheduler(opt, dual_optimizers):
#     dual_scheduler = []
#     for i in range(len(dual_optimizers)):
#         scheduler = lrs.CosineAnnealingLR(
#             dual_optimizers[i],
#             float(opt.epochs),
#             eta_min=opt.eta_min
#         )
#         dual_scheduler.append(scheduler)
#
#     return dual_scheduler


# path_haze = '/home/liuye/Desktop/RCAN-master/RCAN_TrainCode/experiment/test/results/tensor([0])haze.png'
# path_gt = '/home/liuye/Desktop/RCAN-master/RCAN_TrainCode/experiment/test/results/tensor([0])gt.png'
# haze = misc.imread(path_haze)
# gt = misc.imread(path_gt)
# haze = np.swapaxes(haze, 0, 2)
# gt = np.swapaxes(gt, 0, 2)
#
# haze = np.swapaxes(haze, 1, 2)
# gt = np.swapaxes(gt, 1, 2)
#
# haze = torch.from_numpy(haze)
# gt = torch.from_numpy(gt)
# haze, gt = prepare([haze, gt])
# print(calc_ssim(haze, gt))
def init_model(args):

    if args.model.find('DRN') >= 0:
        args.n_blocks = 40
        args.n_feats = 32
    if args.model.find('DRN_v1') >= 0:
        args.n_blocks = 20
        args.n_feats = 32
    if args.model.find('DRND')>=0:
        args.n_feats = 32
    if args.model.find('DRND_v2')>=0:
        args.n_feats = 32
    if args.model.find('DRND_v3') >= 0:
        args.n_feats = 32
    if args.model.find('DRND_v4') >= 0:
        args.n_feats = 32



from PIL import Image
import itertools
from torch.utils.data.sampler import Sampler

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        # print(len(self.secondary_indices))
        # print(len(self.primary_indices))
        # print(self.primary_batch_size)
        # print(self.secondary_batch_size)
        # assert len(self.primary_indices) >= self.primary_batch_size > 0
        # assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
