# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 20:52:34 2018

@author: SENSETIME\yuxian
"""
import os
import random
import numpy as np
from PIL import Image


def filter_50percent_pixel(mask):
    
    if np.sum(mask==0.0)/float(mask.size)  > 0.5:   #tumor region 比例小 
        tumor = False
    else:               #tumor region 比例大
        tumor = True
    
    return tumor
    
'''1024  png'''
#    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/coords/normal_train.txt'
#    save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_NORMAL_TRAIN/mask_HZQ_png/'    
#    
#    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/coords/tumor_valid.txt'
#    save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_TUMOR_VALID/mask_HZQ_png/'
#    
#    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/coords/tumor_train.txt'
#    save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_TUMOR_TRAIN/mask_HZQ_png/'
#
#    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/coords/normal_valid.txt'
#    save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_NORMAL_VALID/mask_HZQ_png/'    
        
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
  
#/lustre/yuxian/Data_t1/pytorch-CycleGAN-and-pix2pix-master/datasets/Pathology/Camelyon16/txt_png_mask
  
    '''Shuffle lines in txt'''
#    path = '/mnt/lustre/yuxian/Data_t1/pytorch-CycleGAN-and-pix2pix-master/datasets/Pathology/Camelyon16/txt_png_mask/'
#    Train_tumor = 'Train_tumor.txt'
#    Train_normal = 'Train_normal.txt'
#    Valid_tumor = 'Valid_tumor.txt'
#    Valid_normal = 'Valid_normal.txt'
#     
#    Train_save = open("/mnt/lustre/yuxian/Data_t1/pytorch-CycleGAN-and-pix2pix-master/datasets/Pathology/train/A/Train_1v1_0_1024_A.txt",'w')
#    Valid_save = open("/mnt/lustre/yuxian/Data_t1/pytorch-CycleGAN-and-pix2pix-master/datasets/Pathology/test/A/Val_1v1_0_1024_A.txt",'w')
#    Train_lines=[]
#    Valid_lines=[]
#    
#    with open(path+Train_tumor, 'r') as infile:        
#        for line in infile:
#            Train_lines.append(line)
#    infile.close()
#    
#    with open(path+Train_normal, 'r') as infile:        
#        for line in infile:
#            Train_lines.append(line)
#    infile.close()
#    
#    random.shuffle(Train_lines)
#    
#    print('length of txt:', len(Train_lines))
#    for idx in range(len(Train_lines)):
#        Train_save.write(Train_lines[idx])
#        if idx%50==0:
#            print(Train_lines[idx])
#            
#    Train_save.close()
#  
#    
#    with open(path+Valid_tumor, 'r') as infile:        
#        for line in infile:
#            Valid_lines.append(line)
#    infile.close()
#    
#    with open(path+Valid_normal, 'r') as infile:        
#        for line in infile:
#            Valid_lines.append(line)
#    infile.close()
#    
#    random.shuffle(Valid_lines)
#    
#    print('length of txt:', len(Valid_lines))
#    for idx in range(len(Valid_lines)):
#        Valid_save.write(Valid_lines[idx])
#        if idx%50==0:
#            print(Valid_lines[idx])
#            
#    Valid_save.close()
    
'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
  
# 像素级别，先判断patch中tumor region的比例，划分为两类    
    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_TUMOR_TRAIN/mask_HZQ_png/'
    patch_folder = '/mnt//lustrenew/yuxian/Code/NCRF-master/Data/resample/PATCHES_TUMOR_TRAIN/'  
    
#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_TUMOR_VALID/mask_HZQ_png/'
#    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_TUMOR_VALID/'  
    
    mask_files = os.listdir(mask_folder)
    
    tumor_list = []
    normal_list = []
    for file_ in mask_files:
        if '.png' in file_:
            mask = Image.open(os.path.join(mask_folder, file_))
            
            if filter_50percent_pixel(np.asarray(mask)):#tumor region
                tumor_list.append(os.path.join(mask_folder, file_))
            else:
                normal_list.append(os.path.join(mask_folder, file_))

    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_NORMAL_TRAIN/mask_HZQ_png/'
    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_NORMAL_TRAIN/'    

#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_NORMAL_VALID/mask_HZQ_png/'
#    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_NORMAL_VALID/'     

    mask_files = os.listdir(mask_folder)
    for file_ in mask_files:
        if '.png' in file_:
            normal_list.append(os.path.join(mask_folder, file_))
#txt_png_mask
#txt_jpeg_mask
    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_png_mask/pixel/Train_tumor.txt","w") as f: 
#    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_png_mask/pixel/Valid_tumor.txt","w") as f: 
        random.shuffle(tumor_list)
        for _file in range(len(tumor_list)):
            
            filename = tumor_list[_file].split('/')[-1][:-4]
            list_ = tumor_list[_file].split('/')[:-2]   
            fuhao = '/'
            patchpath = fuhao.join(list_)
#            print(maskpath)

            f.write(patchpath+'/'+filename+'.jpeg,'+tumor_list[_file]+'\n')   
            print(patchpath+'/'+filename+'.jpeg,'+tumor_list[_file])
            
            
    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_png_mask/pixel/Train_normal.txt","w") as f:
#    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_png_mask/pixel/Valid_normal.txt","w") as f:
        random.shuffle(normal_list)
        for _file in range(len(normal_list)):
            
            filename = normal_list[_file].split('/')[-1][:-4]
            list_ = normal_list[_file].split('/')[:-2]   
            fuhao = '/'
            patchpath = fuhao.join(list_)
#            print(maskpath)

            f.write(patchpath+'/'+filename+'.jpeg,'+normal_list[_file]+'\n')   
            print(patchpath+'/'+filename+'.jpeg,'+normal_list[_file])
        
        f.close()  
    
    print('len(tumor_list)',len(tumor_list))
    print('len(normal_list)',len(normal_list))

# 跟百度的一致，按类别取patch，再拼接  
#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_TUMOR_TRAIN/mask_HZQ_png/'
#    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_TUMOR_TRAIN/'  
#    patch_files = os.listdir(patch_folder)
#    
#    files_list = []
#    for file_ in patch_files:
#        if '.jpeg' in file_:
#            files_list.append(os.path.join(patch_folder, file_))
#
#       
#    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_png_mask/Train_tumor.txt","w") as f:
#
# 
#        random.shuffle(files_list)
#        for _file in range(len(files_list)):
#            filename = files_list[_file].split('/')[-1][:-5]
#            list_ = files_list[_file].split('/')   
#            list_[-1] = 'mask_HZQ_png/'
#            fuhao = '/'
#            maskpath = fuhao.join(list_)
##            print(maskpath)
#
#            f.write(files_list[_file]+','+maskpath+filename+'.png\n')   
#            print(files_list[_file]+','+maskpath+filename)
#        
#        f.close()
#
####################################################################################################
#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_NORMAL_TRAIN/mask_HZQ_png/'
#    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_NORMAL_TRAIN/'  
#    patch_files = os.listdir(patch_folder)
#    
#    files_list = []
#    for file_ in patch_files:
#        if '.jpeg' in file_:
#            files_list.append(os.path.join(patch_folder, file_))
#
#       
#    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_png_mask/Train_normal.txt","w") as f:
# 
#        random.shuffle(files_list)
#        for _file in range(len(files_list)):
#            filename = files_list[_file].split('/')[-1][:-5]
#            list_ = files_list[_file].split('/')   
#            list_[-1] = 'mask_HZQ_png/'
#            fuhao = '/'
#            maskpath = fuhao.join(list_)
##            print(maskpath)
#
#            f.write(files_list[_file]+','+maskpath+filename+'.png\n')   
#            print(files_list[_file]+','+maskpath+filename)
#        
#        f.close()
#        
##        
####################################################################################################
#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_TUMOR_VALID/mask_HZQ_png/'
#    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_TUMOR_VALID/'  
#    patch_files = os.listdir(patch_folder)
#    
#    files_list = []
#    for file_ in patch_files:
#        if '.jpeg' in file_:
#            files_list.append(os.path.join(patch_folder, file_))
#
#    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_png_mask/Valid_tumor.txt","w") as f:
# 
#        random.shuffle(files_list)
#        for _file in range(len(files_list)):
#            filename = files_list[_file].split('/')[-1][:-5]
#            list_ = files_list[_file].split('/')   
#            list_[-1] = 'mask_HZQ_png/'
#            fuhao = '/'
#            maskpath = fuhao.join(list_)
##            print(maskpath)
#
#            f.write(files_list[_file]+','+maskpath+filename+'.png\n')   
#            print(files_list[_file]+','+maskpath+filename)
#        
#        f.close()
#        
####################################################################################################
#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_NORMAL_VALID/mask_HZQ_png/'
#    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/1024/PATCHES_NORMAL_VALID/'  
#    patch_files = os.listdir(patch_folder)
#    
#    files_list = []
#    for file_ in patch_files:
#        if '.jpeg' in file_:
#            files_list.append(os.path.join(patch_folder, file_))
#
#        
#    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_png_mask/Valid_normal.txt","w") as f:
#
#        random.shuffle(files_list)
#        for _file in range(len(files_list)):
#            filename = files_list[_file].split('/')[-1][:-5]
#            list_ = files_list[_file].split('/')   
#            list_[-1] = 'mask_HZQ_png/'
#            fuhao = '/'
#            maskpath = fuhao.join(list_)
##            print(maskpath)
#
#            f.write(files_list[_file]+','+maskpath+filename+'.png\n')   
#            print(files_list[_file]+','+maskpath+filename)
#        
#        f.close()
    
    
   
#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_TRAIN/mask_HZQ/'
#    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_TRAIN/'  
#    patch_files = os.listdir(patch_folder)
#    
#    files_list = []
#    for file_ in patch_files:
#        if '.jpeg' in file_:
#            files_list.append(os.path.join(patch_folder, file_))
#
#       
#    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_jpeg_mask/Train_tumor.txt","w") as f:
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
####################################################################################################
#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_TRAIN/mask_HZQ/'
#    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_TRAIN/'  
#    patch_files = os.listdir(patch_folder)
#    
#    files_list = []
#    for file_ in patch_files:
#        if '.jpeg' in file_:
#            files_list.append(os.path.join(patch_folder, file_))
#
#       
#    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_jpeg_mask/Train_normal.txt","w") as f:
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
##        
####################################################################################################
#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID/mask_HZQ/'
#    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID/'  
#    patch_files = os.listdir(patch_folder)
#    
#    files_list = []
#    for file_ in patch_files:
#        if '.jpeg' in file_:
#            files_list.append(os.path.join(patch_folder, file_))
#
#    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_jpeg_mask/Valid_tumor.txt","w") as f:
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
#        f.close()#
#        
####################################################################################################
#    mask_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_VALID/mask_HZQ/'
#    patch_folder = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_VALID/'  
#    patch_files = os.listdir(patch_folder)
#    
#    files_list = []
#    for file_ in patch_files:
#        if '.jpeg' in file_:
#            files_list.append(os.path.join(patch_folder, file_))
#
#        
#    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_jpeg_mask/Valid_normal.txt","w") as f:
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