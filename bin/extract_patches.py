import glob
import os

import cv2
import numpy as np
import random

from wsi_ops import PatchExtractor 
from wsi_ops import WSIOps 

from multiprocessing import Pool, Value, Lock

DATA_DIR = '/mnt/lustre/share/CAMELYON16/'

WSI_TRAIN = '/mnt/lustre/yuxian/Code/NCRF-master/Data/WSI_TRAIN'
WSI_VALID = '/mnt/lustre/yuxian/Code/NCRF-master/Data/WSI_VAL'


#TUMOR_WSI_PATH = DATA_DIR + 'training/tumor'
#NORMAL_WSI_PATH = DATA_DIR + 'training/normal'

TUMOR_WSI_PATH = '/mnt/lustre/yuxian/Code/NCRF-master/Data/WSI_tumor'
NORMAL_WSI_PATH = '/mnt/lustre/yuxian/Code/NCRF-master/Data/WSI_normal'
TUMOR_MASK_PATH = '/mnt/lustre/yuxian/Code/NCRF-master/Data/mask_training_cv2_level7'

SAVE_DIR = '/mnt/lustre/yuxian/Code/NCRF-master/Data/new_sample/'
PATCHES_TRAIN_NEGATIVE_PATH = SAVE_DIR + 'PATCHES_NORMAL_TRAIN'
PATCHES_TRAIN_POSITIVE_PATH = SAVE_DIR + 'PATCHES_TUMOR_TRAIN'
PATCHES_VALIDATION_NEGATIVE_PATH = SAVE_DIR + 'PATCHES_NORMAL_VALID'
PATCHES_VALIDATION_POSITIVE_PATH = SAVE_DIR + 'PATCHES_TUMOR_VALID'

NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX = 100
NUM_POSITIVE_PATCHES_FROM_EACH_BBOX = 500
PATCH_INDEX_NEGATIVE = 220000#700000
PATCH_INDEX_POSITIVE = 220000

PATCH_NORMAL_PREFIX = 'normal_'
PATCH_TUMOR_PREFIX = 'tumor_'
PATCH_AUG_NORMAL_PREFIX = 'aug_false_normal_'
PATCH_AUG_TUMOR_PREFIX = 'aug_false_tumor_'
PREFIX_SHARD_TRAIN = 'train'
PREFIX_SHARD_AUG_TRAIN = 'train-aug'
PREFIX_SHARD_VALIDATION = 'validation'

TUMOR_PROB_THRESHOLD = 0.90
PIXEL_WHITE = 255
PIXEL_BLACK = 0
PATCH_SIZE = 768#256


count = Value('i', 0)
lock = Lock()
    
        
        
def get_bbox(cont_img, rgb_image=None):
    _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb_contour = None
    if np.size(rgb_image)!=1:
        rgb_contour = rgb_image.copy()
        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    return bounding_boxes, rgb_contour
         
def find_roi_bbox_tumor_gt_mask(mask_image):
#    mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    mask = mask_image
    bounding_boxes, _ = get_bbox(np.array(mask))
    return bounding_boxes
        
def find_roi_bbox(rgb_image):
    # hsv -> 3 channel
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    # mask -> 1 channel
    mask = cv2.inRange(hsv, lower_red, upper_red)

    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    bounding_boxes, rgb_contour = get_bbox(image_open, rgb_image=rgb_image)
    return bounding_boxes, rgb_contour, image_open

def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename
    
def extract_positive_patches_from_tumor_wsi(patch_index, augmentation=False):
    wsi_paths = glob.glob(os.path.join(TUMOR_WSI_PATH, 'tumor*.tif'))
    wsi_paths.sort()
    mask_paths = glob.glob(os.path.join(TUMOR_MASK_PATH, 'tumor*.npy'))
    mask_paths.sort()

    image_mask_pair = zip(wsi_paths, mask_paths)
    image_mask_pair = list(image_mask_pair)
#     image_mask_pair = image_mask_pair[67:68]

    patch_save_dir = PATCHES_TRAIN_POSITIVE_PATH
    patch_prefix = PATCH_TUMOR_PREFIX

    for image_path, mask_path in image_mask_pair:
#    for image_path in image_mask_pair:
        wsi_name = get_filename_from_path(image_path)
        print('extract_positive_patches_from_tumor_wsi(): %s' % wsi_name)
#        wsi_image, _, _, tumor_gt_mask, level_used = wsi_ops.read_wsi_tumor(image_path, mask_path)
        wsi_image, _, tumor_gt_mask, level_used = WSIOps.read_wsi(image_path, mask_path, False)
                                 
#        wsi_image, tumor_gt_mask, level_used = wsi_ops.read_wsi_tumor(image_path, mask_path)
#        wsi_image, rgb_image, _, tumor_gt_mask, level_used = wsi_ops.read_wsi_tumor(image_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

        bounding_boxes = find_roi_bbox_tumor_gt_mask(np.array(tumor_gt_mask))

#        txt_list = []
        patch_index = PatchExtractor.extract_positive_patches_from_tumor_region(wsi_image, np.array(tumor_gt_mask),
                                                                                 level_used, bounding_boxes,
                                                                                 patch_save_dir, patch_prefix,
                                                                                 patch_index,wsi_name)
        
        print('Positive patch count: %d' % (patch_index))
#        txt_list.append(txt_list_item)
        wsi_image.close()



    return patch_index#,txt_list


def extract_negative_patches_from_tumor_wsi(patch_index, augmentation=False):
#    wsi_paths = glob.glob(os.path.join(WSI_VALID, 'tumor*.tif'))
    wsi_paths = glob.glob(os.path.join(TUMOR_WSI_PATH, 'tumor*.tif'))
    wsi_paths.sort()
    mask_paths = glob.glob(os.path.join(TUMOR_MASK_PATH, '*.npy'))
    mask_paths.sort()
    print('hello')
    print(len(wsi_paths))
    print(len(mask_paths))
    image_mask_pair = zip(wsi_paths, mask_paths)
    image_mask_pair = list(image_mask_pair)
    # image_mask_pair = image_mask_pair[67:68]

    patch_save_dir = PATCHES_TRAIN_NEGATIVE_PATH
    patch_prefix = PATCH_NORMAL_PREFIX
    for image_path, mask_path in image_mask_pair:
#    for image_path in image_mask_pair:
        wsi_name = get_filename_from_path(image_path)
        
        print('extract_negative_patches_from_tumor_wsi(): %s' % wsi_name)
#        print('image path ...', image_path)
        wsi_image, rgb_image, tumor_gt_mask, level_used = WSIOps.read_wsi(image_path, mask_path)#
#        wsi_image, rgb_image, _, tumor_gt_mask, level_used = wsi_ops.read_wsi_tumor(image_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

        bounding_boxes, _ , image_open = find_roi_bbox(np.array(rgb_image))

#        txt_list = []
        patch_index = PatchExtractor.extract_negative_patches_from_tumor_wsi(wsi_image, np.array(tumor_gt_mask),
                                                                              image_open, level_used,
                                                                              bounding_boxes, patch_save_dir,
                                                                              patch_prefix,
                                                                              patch_index, wsi_name)

#        write_to_txt(txt_list_item)
        print('Negative patches count: %d' % (patch_index))
#        txt_list.append(txt_list_item)
        wsi_image.close()

    return patch_index#, txt_list

def extract_negative_patches_from_normal_wsi(patch_index, augmentation=False):
    """
    Extracted up to Normal_060.

    :param wsi_ops:
    :param patch_extractor:
    :param patch_index:
    :param augmentation:
    :return:
    """
#    wsi_paths = glob.glob(os.path.join(WSI_VALID, 'normal*.tif'))
    wsi_paths = glob.glob(os.path.join(NORMAL_WSI_PATH, 'normal*.tif'))
#    wsi_paths.shuffle()

#    wsi_paths = wsi_paths[61:]

    patch_save_dir = PATCHES_VALIDATION_NEGATIVE_PATH
    patch_prefix = PATCH_NORMAL_PREFIX
    for image_path in wsi_paths:
        
        wsi_name = get_filename_from_path(image_path)
        print('extract_negative_patches_from_normal_wsi(): %s' % wsi_name)
#        wsi_image, rgb_image, level_used = wsi_ops.read_wsi_normal(image_path)
        mask_path = []
        wsi_image, rgb_image, _, level_used = WSIOps.read_wsi(image_path, mask_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

        bounding_boxes,  _ , image_open = find_roi_bbox(np.array(rgb_image))

#        txt_list = []
        patch_index = PatchExtractor.extract_negative_patches_from_normal_wsi(wsi_image, image_open,
                                                                               level_used,
                                                                               bounding_boxes,
                                                                               patch_save_dir, patch_prefix,
                                                                               patch_index, wsi_name)

        print('Negative patches count: %d' % (patch_index))
#        txt_list.append(txt_list_item)

        wsi_image.close()

    return patch_index#,txt_list
###

def extract_patches():
    patch_index_positive = PATCH_INDEX_POSITIVE
    patch_index_negative = PATCH_INDEX_NEGATIVE#220000
#
    '''negative'''
    patch_index_negative = extract_negative_patches_from_tumor_wsi(patch_index_negative)
    patch_index_negative = extract_negative_patches_from_normal_wsi(patch_index_negative)    
    print('patch_index_negative - PATCH_INDEX_NEGATIVE:%d-%d'% (patch_index_negative,PATCH_INDEX_NEGATIVE))
    
    '''positive'''
#    patch_index_positive = extract_positive_patches_from_tumor_wsi(0)
#    print('patch_index_positive - PATCH_INDEX_POSITIVE:%d-%d'% (patch_index_positive,PATCH_INDEX_POSITIVE))
#
#def extract_patches_augmented(ops, pe):
#    patch_index_positive = PATCH_INDEX_POSITIVE
#    patch_index_negative = PATCH_INDEX_NEGATIVE
#    # index - 500000
#    # index - 700000 -> remove wrong false positives
##    patch_index_negative = extract_patches_from_heatmap_false_region_tumor(ops, pe, patch_index_negative,
##                                                                           augmentation=True)
#    # index - 600000
#    # patch_index_negative = extract_patches_from_heatmap_false_region_normal(ops, pe, patch_index_negative,
#    #                                                                         augmentation=True)
#    # patch_index_negative = extract_negative_patches_from_tumor_wsi(ops, pe, patch_index_negative, augmentation=True)
#    # extract_negative_patches_from_normal_wsi(ops, pe, patch_index_negative, augmentation=True)
#    # extract_positive_patches_from_tumor_wsi(ops, pe, patch_index_positive, augmentation=True)


if __name__ == '__main__':
#    extract_patches_augmented(WSIOps(), PatchExtractor())
     extract_patches()

#     count = Value('i', 0)
#     lock = Lock()
#     pool = Pool(5)
#     pool.map(extract_patches)
