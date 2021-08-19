import torch.utils.data as data
# from PIL import Image
import os
import os.path
import numpy as np
import glob
import cv2
import random
from data import common
IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG', '.mat',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.h5'
]

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def is_jpg_file(filename):
  return filename.endswith('.jpg')
def is_label_file(filename):
  return filename.endswith('.png')
def is_unlabel_file(filename):
  return filename.endswith('.jpeg')
def make_dataset(dirs):
  images = []
  if isinstance(dirs, list):
     for i, dir in enumerate(dirs):
        if not os.path.isdir(dir):
            raise Exception('Check dataroot')
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(dir, fname)
                    item = path
                    images.append(item)
  else:
      for root, _, fnames in sorted(os.walk(dirs)):
          for fname in fnames:
              if is_image_file(fname):
                  path = os.path.join(root, fname)
                  images.append(path)
  return images

def get_patch(img_in, img_tar,trans_map, patch_size):
    ih, iw = img_in.shape[:2]
    p = patch_size
    ix = random.randrange(0, iw - p + 1)
    iy = random.randrange(0, ih - p + 1)
    img_in = img_in[iy:iy + p, ix:ix + p, :]
    img_tar = img_tar[iy:iy + p, ix:ix + p, :]
    trans_map = trans_map[iy:iy + p, ix:ix + p, :]
    return img_in, img_tar, trans_map

def get_patch_da(img_in, patch_size):
    w_total = img_in.shape[1]
    iw = int(w_total / 2)
    ih = img_in.shape[0]
    p = patch_size

    ix = random.randrange(0, iw - p + 1)
    iy = random.randrange(0, ih - p + 1)

    img = img_in[iy:iy + p, ix:ix + p, :]
    img_tar = img_in[iy:iy + p, iw+ix:iw+ix + p, :]
    return img, img_tar

class new_data(data.Dataset):
  def __init__(self, opt, dataroot, seed=None):

    self.dir = dataroot
    self.transform_flip = opt.flip
    self.size = opt.fineSize
    self.unlb_root = opt.unlb_dataroot
    # self.dataroot_da = '/media/ext2/liuye/dataset/domain/train/train'
    if seed is not None:
      np.random.seed(seed)
    # self.paths = sorted(make_dataset(os.path.join(self.dir, 'haze')))
    self.paths = sorted(make_dataset([os.path.join(opt.dataroot, 'haze'), opt.unlb_dataroot]))
    # self.paths = sorted(make_dataset([os.path.join(opt.dataroot, 'haze'), self.dataroot_da]))
    self.GT = os.path.join(opt.dataroot, 'gt')
    # self.GT = [os.path.join(opt.dataroot, 'gt'), self.dataroot_da]
    self.trans = os.path.join(opt.dataroot, 'trans')

  def __getitem__(self, index):
      input_path = self.paths[index]
      # try:
      if is_label_file(input_path):
          filename = os.path.basename(input_path).split('_')[0]
          GT_path = os.path.join(self.GT, filename+'.png')
          # GT_path = os.path.join(self.GT[0], filename + '.png')
          trans_path = os.path.join(self.trans, filename + '.png')
          # print(GT_path)
          haze_image = cv2.imread(input_path).astype("float") / 255.0
          GT = cv2.imread(GT_path).astype("float") / 255.0
          trans = cv2.imread(trans_path).astype("float") / 255.0
          ato = float(os.path.basename(input_path).split('_')[1])
          ato = ato * np.ones((self.size, self.size, 3))
          haze_image, GT, trans = get_patch(haze_image, GT, trans, self.size)
      elif is_jpg_file(input_path):
          AB = cv2.imread(input_path).astype("float") / 255.0
          haze_image, GT = get_patch_da(AB, self.size)
          trans = np.zeros((self.size, self.size, 3))

      else:
          haze_image = cv2.imread(input_path)
          haze_image = haze_image.astype("float") / 255.0  # 注意需要先转化数据类型为float, (0,1)
          haze_image = cv2.resize(haze_image, dsize=(self.size, self.size), interpolation=cv2.INTER_CUBIC)
          trans = np.zeros((self.size, self.size, 3))
          ato = np.zeros((self.size, self.size, 3))
          GT = np.zeros((self.size, self.size, 3))

      haze_image, GT, trans = self.transform(haze_image, GT, trans)

      if self.transform_flip == 1 and random.random() < 0.5:
          haze_image, GT, trans = self.flip(haze_image, GT, trans)
      gt = [haze_image.astype(np.float32), GT.astype(np.float32), trans.astype(np.float32), ato.astype(np.float32)]
      return gt

  def __len__(self):
    train_lb_list=glob.glob(os.path.join(self.dir, 'haze')+'/*png')
    # train_da_list=glob.glob(self.dataroot_da +'/*jpg')
    train_unlb_list = glob.glob(self.unlb_root + '/*jpeg')
    return len(train_lb_list) + len(train_unlb_list)

  def transform(self, haze_image, GT, trans_map):
       haze_image = np.swapaxes(haze_image, 0, 2)
       trans_map = np.swapaxes(trans_map, 0, 2)
       GT = np.swapaxes(GT, 0, 2)

       haze_image = np.swapaxes(haze_image, 1, 2)
       trans_map = np.swapaxes(trans_map, 1, 2)
       GT = np.swapaxes(GT, 1, 2)
       return haze_image, GT, trans_map

  def flip(self, haze_image, GT, trans_map):
      haze_image = np.flip(haze_image, 2).copy()
      trans_map = np.flip(trans_map, 2).copy()
      GT = np.flip(GT, 2).copy()
      return haze_image, GT, trans_map