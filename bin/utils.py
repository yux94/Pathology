# useful small tools for data processing, data loading, visualization, and others
# <huangxiaodi@sensetime.com> 2018-03-13
# based by works from <lijiahui@sensetime.com>

import sys
#sys.path.append('..')
#sys.path.append('../utils/')
import os
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

#========================================================================
#              Data preprocessing & Data Loading
#========================================================================
def normalize(x):
    return x.astype(np.float) / 255.0

def resize(img, imgW, imgH):
    return img.resize((imgW, imgH), Image.ANTIALIAS)

def mean(img, mean_value):
    img[...,0] -= mean_value[0]
    img[...,1] -= mean_value[1]
    img[...,2] -= mean_value[2]
    return img

def load_image(file, imgW, imgH, mean_value = [0,0,0]):
    # for image, should follow steps:
    #   load -> Image resize(ANTIALIAS) -> to numpy -> mean -> normalize
    img =  Image.open(file)

    # Should convert Image to numpy array for following processing
    img = img.resize((imgW,imgH),Image.ANTIALIAS)
    #img = normalize(mean(img, mean_value))
    return img

def load_mask(file, imgW, imgH):
    # for mask image, should follow steps:
    #   load -> Image resize -> to numpy to *255 -> image
    mask = Image.open(file).convert('L')
    mask = mask.resize((imgW,imgH),Image.NEAREST)

    # mask is numpy arrary, -> torch.from_numpy -> tensor
    #mask = (np.array(mask) > 0).astype(np.uint8) # if mask is 0,1 image

    # mask is a image

    return mask


#========================================================================
#               Train & Test
#========================================================================
class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. and normal numpy float"""

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()
        else:
            count = 1
            v = v

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def length(self):
        return self.n_count

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

class sumer(object):
    '''Comput sum for numpy float'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def length(self):
        return self.n_count

    def val(self):
        return self.sum

    def add(self,v):
        self.n_count += 1
        self.sum += float(v)


def pixel_wise_accuracy(pred, label):
    '''accuray between two numpy arrays => two single images couple'''
    if not pred.shape == pred.shape:
        print("input shapes are different")
        return -1

    pred = (pred.astype(np.float) > 0.5).astype(np.int)
    label = (label.astype(np.float) > 0.5).astype(np.int)
    acc = float(np.count_nonzero(label==pred))/label.size
    return acc

def convert2hms(time):
    m,s = divmod(time, 60)
    h,m = divmod(m,60)
    return h,m,s

def pixel_wise_accuracy_tensor(pred, label):
    '''pred and label are two torch tensors'''
    acc = float(label.eq(pred).sum()) / float(label.numel())
    return acc

def dice_score_tensor(pred, label):
    '''pred and label are two torch tensors'''
    inter = torch.dot(pred.view(-1),label.view(-1)) + 0.0001
    union = pred.sum() + label.sum() + 0.0001
    dice = 2*inter / union
    return dice

def nearest256(number):
    '''find the nearest 256 number'''
    times = int(number/256)
    if times == 0:
        return 256
    else:
        return 256*times
