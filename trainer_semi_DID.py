#-- coding:UTF-8 --
import torch
import utility
import numpy as np
import datetime
from tqdm import tqdm
import os
import math
from utility import *
from random import uniform
class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class Trainer_semi_DID():
    def __init__(self, opt, loader, my_model, ema_model, my_loss, ckp):
        self.opt = opt
        self.ckp = ckp
        self.epoch = 0
        self.loader_train = loader
        self.loader_test = loader
        self.model = my_model
        self.ema_model = ema_model
        self.loss = torch.nn.L1Loss().cuda()
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.dual_flag = opt.dual_flag
        self.consistency = opt.consistency
        self.consistency_rampup = opt.consistency_rampup
        self.labeled_bs = opt.labeled_bs
        self.error_last = 1e8
        self.curiter = 0
        self.max_iters = self.opt.epochs * len(self.loader_train) // self.opt.nAveGrad
        self.dir = './experiment/' + self.opt.save
        self.log_path = os.path.join('./experiment/', self.opt.save, str(datetime.datetime.now()) + '.txt')
        self.log_test = os.path.join('./experiment/', self.opt.save,
                                     str(datetime.datetime.now()) + '_psnr.txt')

    def train(self, epoch):
        aveGrad = 0
        self.epoch = epoch
        self.model.train()
        self.ema_model.train()
        # epoch = self.scheduler.last_epoch + 2
        timer_data, timer_model = utility.timer(), utility.timer()
        sup_loss_dehaze_record = AvgMeter()
        sup_loss_t_record, con_loss_dehaze_record, con_loss_t_record = AvgMeter(), AvgMeter(), AvgMeter()
        for batch, gt in enumerate(self.loader_train):
            input, gt, trans, ato = gt
            input, gt, trans, ato = self.prepare([input, gt, trans, ato])

            timer_data.hold()
            timer_model.tic()

            self.optimizer.param_groups[0]['lr'] = self.opt.lr * (
                         1 - (self.epoch / self.opt.epochs)) ** self.opt.power
                        # 1 - (self.curiter / self.opt.max_iters)) ** self.opt.power
            self.optimizer.zero_grad()
            if not self.dual_flag:
                noise = torch.clamp(torch.randn_like(input[self.labeled_bs:]) * 0.1, -0.1, 0.1)
                # ema_input = self.augmentation(input[self.labeled_bs:], trans[self.labeled_bs:])
                ema_input = input[self.labeled_bs:]
                ema_input = ema_input.to('cuda')
                #label数据
                predict1, predict2 = self.model(input)
                with torch.no_grad():
                    ema_predict1, ema_predict2 = self.ema_model(ema_input)
                loss_dehaze = self.loss(predict1[0][0:self.labeled_bs], gt[0:self.labeled_bs]) \
                              + self.loss(predict2[0][0:self.labeled_bs], gt[0:self.labeled_bs])

                loss_trans = self.loss(predict1[1][0:self.labeled_bs], trans[0:self.labeled_bs]) \
                             + self.loss(predict2[1][0:self.labeled_bs], trans[0:self.labeled_bs])

                loss_ato = self.loss(predict1[2][0:self.labeled_bs], ato[0:self.labeled_bs]) \
                             + self.loss(predict2[2][0:self.labeled_bs], ato[0:self.labeled_bs])

                loss_rec = self.loss(predict1[3][0:self.labeled_bs], input[0:self.labeled_bs]) \
                           + self.loss(predict2[3][0:self.labeled_bs], input[0:self.labeled_bs])

                sup_loss = loss_dehaze + 0.3*loss_trans + 0.1*loss_ato + 0.1*loss_rec

                con_loss_dehaze = self.loss(predict1[0][self.labeled_bs:], ema_predict1[0]) + \
                                  self.loss(predict2[0][self.labeled_bs:], ema_predict2[0])

                con_loss_trans = self.loss(predict1[1][self.labeled_bs:], ema_predict1[1]) + \
                                 self.loss(predict2[1][self.labeled_bs:], ema_predict2[1])

                con_loss_ato = self.loss(predict1[2][self.labeled_bs:], ema_predict1[2])+\
                               self.loss(predict2[2][self.labeled_bs:], ema_predict2[2])

                con_loss_rec = self.loss(predict1[3][self.labeled_bs:], ema_predict1[3]) + \
                               self.loss(predict2[3][self.labeled_bs:], ema_predict2[3])

                con_loss = con_loss_dehaze + 0.3*con_loss_trans + 0.1*con_loss_ato + 0.1*con_loss_rec
                consistency_weight = self.get_current_consistency_weight(epoch)
                loss = sup_loss + consistency_weight * con_loss
            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()
                aveGrad += 1
                sup_loss_dehaze_record.update(loss_dehaze.data, self.opt.batchSize)
                sup_loss_t_record.update(loss_trans.data, self.opt.batchSize)
                con_loss_dehaze_record.update(con_loss_dehaze.data, self.opt.batchSize)
                # con_loss_t_record.update(con_loss_trans.data, self.opt.batchSize)
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
            if aveGrad % (self.opt.nAveGrad / self.opt.batchSize) == 0:
                self.optimizer.step()
                self.update_ema_variables(self.model, self.ema_model, self.opt.ema_decay, self.curiter)
                self.curiter = self.curiter + 1
                aveGrad = 0


            timer_model.hold()
            if (batch + 1) % self.opt.print_every == 0:
                log = '[epoch %d],[loss_dehaze_s %.5f],[loss_t_s %.5f],[loss_dehaze_con %.5f], [lr %.13f]' % \
                      (self.epoch, sup_loss_dehaze_record.avg, sup_loss_t_record.avg, con_loss_dehaze_record.avg,
                       self.optimizer.param_groups[0]['lr'])
                       # self.scheduler.get_lr()[0])
                print(log)
                open(self.log_path, 'a').write(log + '\n')

            timer_data.tic()

        snapshot_name = 'epoch%d' % (epoch)
        torch.save(self.model.state_dict(), os.path.join(self.dir, 'model', snapshot_name + '.pt'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.dir, 'model', snapshot_name + '_optim.pt'))

    def test(self, count):
        self.model.eval()
        avgPSNR = 0.0
        avgSSIM = 0.0
        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (haze, gt, trans, ato, index) in enumerate(tqdm_test):
                no_eval = (gt.nelement() == 1)
                if not no_eval:
                    haze, gt, trans, ato = self.prepare([haze, gt, trans, ato])
                else:
                    haze = self.prepare(haze)

                _, predict= self.model(haze)
                if isinstance(predict, list):
                    dehaze = predict[0]

                dehaze = quantize(dehaze)
                gt = 255 * gt
                dehaze = 255 * dehaze
                avgSSIM += SSIM(dehaze, gt)
                avgPSNR += PSNR(dehaze, gt)

            log = '[epoch:%d]\t[PSNR: %.4f]\t[SSIM: %.4f]\n'\
                %(count, avgPSNR / len(self.loader_test), avgSSIM / len(self.loader_test))
            print(log)
            open(self.log_test, 'a').write(log + '\n')

    def augmentation(self, input, trans):
        input= input.cpu()
        trans = trans.cpu()
        for i in range(0,input.shape[0]):
            haze = input[i]
            depth = trans[i]
            # beta = uniform(0.4, 1.6)
            beta = uniform(0, 0.4)
            maxhazy = depth.max()
            depth = depth / (maxhazy)
            transmission = np.exp(-beta * depth)

            # a = 1 - 0.5 * uniform(0, 1)
            a = 0.2 * uniform(0, 1)
            m = depth.shape[1]
            n = depth.shape[2]
            rep_atmosphere = np.tile(np.full([3, 1, 1], a), [1, m, n])

            haze_image = haze * transmission + torch.Tensor(rep_atmosphere) * (1 - transmission)
            input[i] = haze_image
        return input

    def step(self):
        self.scheduler.step()

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        def _prepare(tensor):
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.epoch
            return epoch >= self.opt.epochs

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

