# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 16:35:31 2018

@author: SENSETIME\yuxian
"""

# -*- coding=utf-8 -*-
# do model evaluation
# <huangxiaodi@sensetime.com> 2018-03-27

import sys
sys.path.append('..')
sys.path.append('../process_data/')
#from kfb_linux.kfbslide import KfbSlide
#from process_data.extraction_patch import Mask
#from process_data.extraction_tissue import extraction_tissue
#from process_data.data_utils import get_region
import openslide
from scipy import ndimage as nd
from PIL import Image

import os, glob, re, shutil
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from random import shuffle
from PIL import Image
#import cv2
import matplotlib.pyplot as plt
import time
import pandas as pd
from skimage import measure

from utils import averager, sumer, pixel_wise_accuracy, resize, load_image, load_mask, pixel_wise_accuracy_tensor, dice_score_tensor, convert2hms
from matrix import pixel_wise_accuracy_numpy, dice_score_numpy, tp_fp_tn_fn,pixel_wise_accuracy_numpy_1

def computeEvaluationMask(maskDIR, level):
    """Computes the evaluation mask.
    
    Args:
        maskDIR:    the directory of the ground truth mask
        level:      The level at which the evaluation mask is made
        
    Returns:
        evaluation_mask
    """
    resolution = 0.243

    '''with tiff mask'''
#    slide = openslide.open_slide(maskDIR)
#    dims = slide.level_dimensions[level]
#    pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')
#    pixelarray = np.array(slide.read_region((0,0), level, dims))[:,:,0]
    

    '''with npy mask'''
    print('loading mask ...:',maskDIR)
    pixelarray = np.load(maskDIR, mmap_mode='r')/255
    
    return pixelarray   
#    return evaluation_mask 
    
    
if __name__ == "__main__":

#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/TEST_MASK'#sys.argv[1]  #ground turth
#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/mask_testing_cv2_level7/'
    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/mask_testing_cv2_level6/'

#    result_folder = '/mnt/lustre/yuxian/Code/NCRF-master/PROBS_MAP_PATH/crf/LEVEL7'#sys.argv[2]
#    result_folder = '/mnt/lustre/yuxian/Code/NCRF-master/PROBS_MAP_PATH/crf/LEVEL6'#sys.argv[2]
#    result_folder = '/mnt/lustrenew/yuxian/Code/NCRF-master/PROBS_MAP_PATH/crf/xiaodi_level6/'  
#    result_folder = '/mnt/lustrenew/yuxian/Code/NCRF-master/PROBS_MAP_PATH/xiaodi_level6_jpeg768_balance2/30epoch/'
    result_folder = '/mnt/lustrenew/yuxian/Code/NCRF-master/PROBS_MAP_PATH/xiaodi_level6_png1024_balance1/'
    
    result_file_list = []
    result_file_list += [each for each in os.listdir(result_folder) if each.endswith('.npy')]
    
    EVALUATION_MASK_LEVEL = 5 # Image level at which the evaluation is done
        
    ground_truth_test = []
#    ground_truth_test += [each[0:8] for each in os.listdir(mask_folder) if each.endswith('.tif')]
    ground_truth_test += [each[0:8] for each in os.listdir(mask_folder) if each.endswith('.npy')]
#    ground_truth_test += [each[0:8] for each in os.listdir(mask_folder) if each.endswith('.jpeg')]
    ground_truth_test = set(ground_truth_test)
    
#    print(ground_truth_test)
    caseNum = 0  
    # init averagers
    test_acc = averager()
    test_acc_0 = averager()
    test_acc_1 = averager()
    test_acc_2 = averager()
    test_acc_2_1 = averager()
    test_dice = averager()
    test_dice_0 = averager()
    test_dice_1 = averager()
    test_dice_2 = averager()

    # init sumers
    test_tp = sumer()
    test_fp = sumer()
    test_tn = sumer()
    test_fn = sumer()

    # init min,max
    test_acc_min = 1.0
    test_acc_max = 0.0
    test_dice_min = 1.0
    test_dice_max = 0.0
    test_dice_0_min = 1.0
    test_dice_0_max = 0.0
    test_dice_1_min = 1.0
    test_dice_1_max = 0.0
    test_dice_2_min = 1.0
    test_dice_2_max = 0.0

    # bad result list
    bad_list = []

    for case in result_file_list:
#        print('Evaluating Performance on image:', case[0:-4])
        sys.stdout.flush()
        predDIR = os.path.join(result_folder, case)
        
        filename = case.split('.')[0].split('/')[-1]
#        print('processing:',filename)
        pred = np.load(predDIR)
#        pred = np.transpose(pred)
        pred = (pred > 0.5).astype(np.uint8)
        
#        maskDIR = os.path.join(mask_folder, filename) + '.tif'
        maskDIR = os.path.join(mask_folder, filename) + '.npy'
#        maskDIR = os.path.join(mask_folder, filename) + '.jpeg'
        if not os.path.exists(maskDIR):
            evaluation_mask =  np.zeros(np.shape(pred),dtype=pred.dtype)
        else:
            evaluation_mask = computeEvaluationMask(maskDIR, EVALUATION_MASK_LEVEL)    
#            evaluation_mask = np.asarray(Image.open(maskDIR))    
                
#        np.save('/mnt/lustre/yuxian/Code/NCRF-master/TEST_MASK/level7/tiff/'+filename,evaluation_mask)    
#        np.save('/mnt/lustre/yuxian/Code/NCRF-master/PROBS_MAP_PATH/PRED_LEVEL7/'+filename,pred)  
        
        print(np.shape(pred))
        print(np.shape(evaluation_mask))
        if(np.shape(pred)==np.shape(evaluation_mask)):
            pass
        else:
            pred = np.transpose(pred)

        print(np.max(pred))
        print(np.max(evaluation_mask))
#        if not np.shape(pred)==np.shape(evaluation_mask):
#            pred = np.transpose(pred)
        tp2,fp2,tn2,fn2 = tp_fp_tn_fn(pred, evaluation_mask)
        acc2 = pixel_wise_accuracy_numpy(pred, evaluation_mask)
        dice2 = dice_score_numpy(pred, evaluation_mask) 
        # average dice score
        if acc2 < 0.5 or dice2 < 0.5:
            bad_list.append([filename,acc2,dice2])
            
        dice2_0 = 0.0   # all 0 patch
        dice2_1 = 0.0   # all 1 patch
        dice2_2 = 0.0   # half patch
        if np.sum(evaluation_mask) == 0.0:#all 0 patch
            evaluation_mask = (evaluation_mask == 0).astype(np.uint8)
            pred = (pred == 0).astype(np.uint8)
            
            acc2_0 = pixel_wise_accuracy_numpy(pred, evaluation_mask)
            
            inter = np.sum(evaluation_mask[pred == 1]) + 0.000001
            union = np.sum(pred) + np.sum(evaluation_mask) + 0.00001
            dice2_0 = 2*inter / union
            
            test_acc_0.add(acc2_0)
            
            test_dice_0.add(dice2_0)
            if test_dice_0_min > dice2_0: test_dice_0_min = dice2_0
            if test_dice_0_max < dice2_0: test_dice_0_max = dice2_0
        elif np.sum(evaluation_mask) == float(evaluation_mask.size):#all 1 patch
                    
            acc2_1 = pixel_wise_accuracy_numpy(pred, evaluation_mask)
            
            inter = np.sum(evaluation_mask[pred == 1]) + 0.000001
            union = np.sum(pred) + np.sum(evaluation_mask) + 0.00001
            dice2_1 = 2*inter / union
            test_acc_1.add(acc2_1)
            test_dice_1.add(dice2_1)
            if test_dice_1_min > dice2_1: test_dice_1_min = dice2_1
            if test_dice_1_max < dice2_1: test_dice_1_max = dice2_1
        else:# half patch
            acc2_2 = pixel_wise_accuracy_numpy(pred, evaluation_mask)         
            acc2_2_1 = pixel_wise_accuracy_numpy_1(pred, evaluation_mask)     
            
            inter = np.sum(evaluation_mask[pred == 1]) + 0.000001
            union = np.sum(pred) + np.sum(evaluation_mask) + 0.00001
            dice2_2 = 2*inter / union
            test_acc_2.add(acc2_2)
            test_acc_2_1.add(acc2_2_1)
            test_dice_2.add(dice2_2)
            if test_dice_2_min > dice2_2: test_dice_2_min = dice2_2
            if test_dice_2_max < dice2_2: test_dice_2_max = dice2_2
        
        print('filename:', filename)
        print('test result: acc: %.6f < %.6f < %.6f, dice: %.6f < %.6f < %.6f'%(test_acc_min, acc2, test_acc_max, test_dice_min, dice2, test_dice_max))

                
        test_tp.add(tp2)
        test_fp.add(fp2)
        test_tn.add(tn2)
        test_fn.add(fn2)
        test_acc.add(acc2)
        test_dice.add(dice2)
        if test_acc_min > acc2: test_acc_min = acc2
        if test_acc_max < acc2: test_acc_max = acc2
        if test_dice_min > dice2: test_dice_min = dice2
        if test_dice_max < dice2: test_dice_max = dice2

    test_TPR = float(test_tp.val() + 0.000001) / (test_tp.val() + test_fn.val() + 0.000001)
    test_TNR = float(test_tn.val() + 0.000001) / (test_tn.val() + test_fp.val() + 0.000001)
    test_PPV = float(test_tp.val() + 0.000001) / (test_tp.val() + test_fp.val() + 0.000001)
    test_FPV = float(test_tn.val() + 0.000001) / (test_tn.val() + test_fn.val() + 0.000001)
    test_ACC = float(test_tp.val() + test_tn.val() + 0.000001) / (test_tp.val() + test_fp.val() + test_tn.val() + test_fn.val() + 0.000001)
    test_F1_Score = 2*test_TPR*test_PPV / (test_TPR + test_PPV)

    print( '====evaluation finished=======\n\n')
    print( 'Test result: acc: %.6f < %.6f < %.6f, dice: %.6f < %.6f < %.6f,\n acc_0(%d):%.6f, acc_1(%d):%.6f, acc_2(%d):%.6f, acc_2_1:%.6f, dice_0(%d): %.6f , dice_1(%d): %.6f , dice_2(%d): %.6f \n, TPR: %.6f, TNR: %.6f, PPV: %.6f, FPV: %.6f, ACC: %.6f, F1_Score: %.6f' % (test_acc_min, test_acc.val(), test_acc_max, test_dice_min, test_dice.val(), test_dice_max, test_dice_0.length(), test_acc_0.val(), test_dice_1.length(), test_acc_1.val(), test_dice_2.length(), test_acc_2.val(), test_acc_2_1.val(), test_dice_0.length(), test_dice_0.val(), test_dice_1.length(), test_dice_1.val(), test_dice_2.length(), test_dice_2.val(), test_TPR, test_TNR, test_PPV, test_FPV, test_ACC, test_F1_Score))
    

