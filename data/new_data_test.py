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
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.h5'
]

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def is_jpg_file(filename):
  return filename.endswith('.jpg')

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

class new_data(data.Dataset):
  def __init__(self, opt, dataroot, seed=None):

    self.dir = dataroot
    self.transform_flip = opt.flip
    self.size = opt.fineSize
    if seed is not None:
      np.random.seed(seed)
    self.input = sorted(make_dataset(os.path.join(self.dir, 'haze')))
    self.GT = os.path.join(self.dir, 'gt')
    self.trans = os.path.join(self.dir, 'trans')

  def __getitem__(self, index):
      input_path = self.input[index]
      filename = os.path.basename(input_path).split('_')[0]
      # print(input_path)
      GT_path = os.path.join(self.GT, filename + '.png')
      trans_path = os.path.join(self.trans, filename + '.png')

      haze_image = cv2.imread(input_path).astype("float") / 255.0
      GT = cv2.imread(GT_path).astype("float") / 255.0
      trans = cv2.imread(trans_path).astype("float") / 255.0

      ato = float(os.path.basename(input_path).split('_')[1])
      ato = ato * np.ones((self.size, self.size, 3))
      haze_image, h_gt, w_gt = self.padding(haze_image)
      # h = haze_image.shape[0]
      # w = haze_image.shape[1]
      haze_image, GT, trans = self.transform(haze_image, GT, trans)
      # haze_image = cv2.resize(haze_image, (320,320))
      cv2.setNumThreads(0)
      cv2.ocl.setUseOpenCL(False)
      return haze_image.astype(np.float32), GT.astype(np.float32), filename, h_gt, w_gt, -1

  def __len__(self):
    train_lb_list=glob.glob(os.path.join(self.dir, 'haze')+'/*png')
    return len(train_lb_list)

  def transform(self, haze_image, GT, trans_map):
       haze_image = np.swapaxes(haze_image, 0, 2)
       trans_map = np.swapaxes(trans_map, 0, 2)
       GT = np.swapaxes(GT, 0, 2)

       haze_image = np.swapaxes(haze_image, 1, 2)
       trans_map = np.swapaxes(trans_map, 1, 2)
       GT = np.swapaxes(GT, 1, 2)
       return haze_image, GT, trans_map

  def padding(self, haze_image):
        h_gt = haze_image.shape[0]
        w_gt = haze_image.shape[1]
        h = int(haze_image.shape[0] / 16 + 1) * 16
        w = int(haze_image.shape[1] / 16 + 1) * 16
        haze_image = cv2.copyMakeBorder(haze_image, 0, h - int(haze_image.shape[0]), 0,
                                           w - int(haze_image.shape[1]), cv2.BORDER_REPLICATE)
        return haze_image, h_gt, w_gt