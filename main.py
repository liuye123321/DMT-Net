import torch
import random
import utility
import data
import model
import loss
from utility import TwoStreamBatchSampler
from option import args
from trainer_semi_DID import Trainer_semi_DID
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
args.manualSeed = random.randint(1, 10000)
count = 1
def create_emamodel(net, ema=True):
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

def getLoader(datasetName, dataroot,batchSize=64, workers=4,
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), split='train', shuffle=True, seed=None):

  #import pdb; pdb.set_trace()
  if datasetName == 'new_data_train':
    from data.new_data_train import new_data as commonDataset
  if datasetName == 'new_data_val':
    from data.new_data_test import new_data as commonDataset

  dataset = commonDataset(args, dataroot)

  return dataset

def relabel_dataset(dataset):
    labeled_idxs = []
    for idx in range(len(dataset.paths)):
        # if dataset.paths[idx].endswith('.h5'):
        if dataset.paths[idx].endswith('.png'):
            labeled_idxs.append(idx)
    unlabeled_idxs = sorted(set(range(len(dataset))) - set(labeled_idxs))
    return labeled_idxs, unlabeled_idxs

if checkpoint.ok:
    dataset_train = getLoader(args.data_train,    # misc.py
                           args.dataroot,
                           args.batchSize,
                           args.workers,
                           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                           split='train',
                           shuffle=True,
                           seed=args.manualSeed)
    dataset_test = getLoader(args.data_test,
                           args.testpath,
                           args.batchSize,
                           args.workers,
                           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                           split='val',
                           shuffle=False,
                           seed=args.manualSeed)
    labeled_idxs, unlabeled_idxs = relabel_dataset(dataset_train)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batchSize, args.batchSize - args.labeled_bs)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=batch_sampler,
        num_workers=int(args.workers))
    # dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchSize, num_workers=int(args.workers), shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=int(args.workers))
    loss = loss.Loss(args, checkpoint) if not args.test_only else None

    if args.model == 'DID':
        net = model.Model_DID(args, checkpoint)
        ema_net = model.Model_DID(args, checkpoint)
        ema_net = create_emamodel(ema_net)
        t = Trainer_semi_DID(args, dataloader_train, net, ema_net, loss, checkpoint)
        test = Trainer_semi_DID(args, dataloader_test, net, ema_net, loss, checkpoint)
    while not t.terminate():
        t.train(count)
        if count % 1 == 0:
            test.test(count)
        count+=1
    checkpoint.done()

