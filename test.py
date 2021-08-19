
import torch
import random
import utility
import data
import model
import loss
from option import args

from trainer_semi_DID import Trainer_semi_DID

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
global args
count = 1
args.testpath ='the path of test datasets'
args.save = 'test'
args.model = 'DID'
# args.dataset = 'new_data_val'
torch.manual_seed(args.seed)
args.test_only = True
args.pre_train = 'pretrained model path'
checkpoint = utility.checkpoint(args)
args.manualSeed = random.randint(1, 10000)
def getLoader(datasetName, dataroot, transform,  batchSize=1, workers=1,
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), split='train', shuffle=True, seed=None):

  if datasetName == 'new_data_train':
    from data.new_data_train import new_data as commonDataset
  if datasetName == 'new_data_val':
    from data.new_data_test import new_data as commonDataset

  dataset = commonDataset(args, dataroot)

  dataloader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batchSize,
                                           shuffle=shuffle,
                                           num_workers=int(workers)
                            )
  return dataloader
if checkpoint.ok:
    # loader = data.Data(args)
    dataloader = getLoader(args.data_test,
                           args.testpath,
                           args.batchSize,
                           args.workers,
                           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                           split='val',
                           shuffle=False,
                           seed=args.manualSeed)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None

    if args.model == 'DID':
        net = model.Model_DID(args, checkpoint)
        t = Trainer_semi_DID(args, dataloader, net, loss, checkpoint)
    t.test(count)

    checkpoint.done()


