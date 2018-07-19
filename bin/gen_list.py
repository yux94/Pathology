# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 20:52:34 2018

@author: SENSETIME\yuxian
"""
import os
import random


'''mask folder'''
#        save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID/mask_HZQ/'
#        save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_TRAIN/mask_HZQ/'
#        save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_VALID/mask_HZQ/'
#        save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_TRAIN/mask_HZQ/'

'''patch folder'''
#        save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID/'
#        save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_TRAIN/'
#        save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_VALID/'
#        save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_TRAIN/'

if __name__ == '__main__':
    
   
#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_TRAIN/mask_HZQ/'
#    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_TRAIN/'  
#    patch_files = os.listdir(patch_folder)
#    
#    files_list = []
#    for file_ in patch_files:
#        if '.jpeg' in file_:
#            files_list.append(os.path.join(patch_folder, file_))
#
#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_TRAIN/mask_HZQ/'
#    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_TRAIN/'  
#    patch_files = os.listdir(patch_folder)
#    
#    for file_ in patch_files:
#        if '.jpeg' in file_:
#            files_list.append(os.path.join(patch_folder, file_))
#
#        
#    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_jpeg_mask/Train.txt","w") as f:
#
# 
#        random.shuffle(files_list)
#        for _file in range(len(files_list)):
#            filename = files_list[_file].split('/')[-1]
#            list_ = files_list[_file].split('/')   
#            list_[-1] = 'mask_HZQ/'
#            fuhao = '/'
#            maskpath = fuhao.join(list_)
##            print(maskpath)
#
#            f.write(files_list[_file]+','+maskpath+filename+'\n')   
#            print(files_list[_file]+','+maskpath+filename)
#        
#        f.close()
#        
    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID/mask_HZQ/'
    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID/'  
    patch_files = os.listdir(patch_folder)
    
    files_list = []
    for file_ in patch_files:
        if '.jpeg' in file_:
            files_list.append(os.path.join(patch_folder, file_))

    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_VALID/mask_HZQ/'
    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_VALID/'  
    patch_files = os.listdir(patch_folder)
    
    for file_ in patch_files:
        if '.jpeg' in file_:
            files_list.append(os.path.join(patch_folder, file_))

        
    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_jpeg_mask/Valid.txt","w") as f:

        random.shuffle(files_list)
        for _file in range(len(files_list)):
            filename = files_list[_file].split('/')[-1]
            list_ = files_list[_file].split('/')   
            list_[-1] = 'mask_HZQ/'
            fuhao = '/'
            maskpath = fuhao.join(list_)
#            print(maskpath)

            f.write(files_list[_file]+','+maskpath+filename+'\n')   
            print(files_list[_file]+','+maskpath+filename)
        
        f.close()