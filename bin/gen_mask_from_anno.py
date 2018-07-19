# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:20:15 2018

@author: SENSETIME\yuxian
"""
import logging
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os, sys, shutil, argparse
import openslide
import json
from scipy import ndimage as ndi
from PIL import Image

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save it in npy format')
parser.add_argument('--wsi_path', default='../../Data/WSI_VAL/', metavar='TIF_PATH', type=str, help='Path to the original tif file')
parser.add_argument('--json_path', default='../../jsons/valid/', metavar='JSON_PATH', type=str, help='Path to annotation json file (created from xml)')
parser.add_argument('--level', default=0, type=int, help='at which WSI level to obtain the mask, default 6')
parser.add_argument('--mask_path', default='../../Data/WSI_VAL/mask_level0_jpeg_gen_from_anno/', metavar='MASK_PNG_PATH', type=str, help='Path to mask png file')

args = parser.parse_args()
path =args.json_path

# print(len(positives))
# print(len(negatives))

def run(args):
    scale = 2**(args.level)
    filenames = os.listdir(args.wsi_path)
    num = 0
    for filename in filenames:
        num +=1
        logging.info('processing {}/{}...'.format(num,len(filenames)))
        
        slide = openslide.OpenSlide(args.wsi_path+filename)
        # note the shape of img_RGB is the transpose of slide.level_dimensions
        img_RGB = np.transpose(
            np.array(slide.read_region((0, 0),args.level, slide.level_dimensions[args.level]).convert('RGB')),
            axes=[1, 0, 2]
        )
        #level 6 mask shape
        mh, mw = img_RGB.shape[0], img_RGB.shape[1]
        mask_arr = np.zeros((mh, mw))
        # print('mask_arr of shape', mask_arr.shape)
        # print('img_RGB is of shape', mask_arr.shape)
        if os.path.exists(path+filename[:-4]+'.json'):

            file = open(path+filename[:-4]+'.json', 'r')
            jdict = json.load(file)
            
            positives =  jdict['positive']
            logging.info('gen mask from anno with {}...'.format(filename))
        
            for i in range(len(positives)):
                outline = np.asarray(positives[i]['vertices']) // scale
                # print(outline)
                x = outline[:,0]
                y = outline[:,1]
                mask_arr[x,y] = 1.0
                
        #not sure if I should do this, but ignore the negetives
        mask_arr = ndi.binary_dilation(mask_arr)
        mask_arr = ndi.binary_fill_holes(mask_arr)
#        plt.figure(figsize=(10,10)) 
#        plt.imshow(t2, cmap='gray')
#        plt.savefig(args.mask_path+filename[:-4])
#        plt.close()
        mask_arr = Image.fromarray((mask_arr*255).astype(np.uint8))        
#        mask_arr = mask_arr.rotate(90) 
#        mask_arr = mask_arr.transpose(Image.FLIP_TOP_BOTTOM)         
        mask_arr.save(args.mask_path+filename[:-4]+'.jpeg')
        
#        img_RGB = Image.fromarray((img_RGB*255).astype(np.uint8))   
#        img_RGB = img_RGB.rotate(90) 
#        img_RGB = img_RGB.transpose(Image.FLIP_TOP_BOTTOM)      
#        img_RGB.save(args.mask_path+filename[:-4]+'_raw.jpeg')
        
        
def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()