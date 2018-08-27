# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:01:44 2018

@author: SENSETIME\yuxian
"""
import os
import staintools
stain_normalizer = staintools.StainNormalizer(method='vahadane')
standardizer = staintools.BrightnessStandardizer()#Brightness Standardization



#patch_dir = 'PATCHES_NORMAL_TRAIN'
#patch_dir = 'PATCHES_TUMOR_TRAIN'
#patch_dir = 'PATCHES_NORMAL_VALID'
patch_dir = 'PATCHES_TUMOR_VALID'

img_dir = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/'+patch_dir
save_dir = '/mnt/lustre/yuxian/Code/NCRF-master/Data/Patch_Stain_Norm/1024/SN/'+patch_dir
img_files = os.listdir(img_dir)

num=0
total=len(img_files)

for img_file in img_files:
    if '.jpeg' in img_file:
        num += 1
        print('%d/%d with img name %s'%(num,total,img_file))
        
        Original_img = staintools.read_image(os.path.join(img_dir, img_file))
        '''光照信息'''
        img_standard = standardizer.transform(Original_img)
        
        img_standard_normalized = stain_normalizer.transform(img_standard)
        
        img_standard_normalized.save(os.path.join(save_dir, img_file))