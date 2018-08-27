# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 15:36:19 2018

@author: SENSETIME\yuxian
"""

import numpy as np
from PIL import Image
import openslide
from scipy import ndimage as nd
from skimage import measure
from skimage.measure import regionprops
#from pyheatmap.heatmap import HeatMap
import matplotlib.pyplot as plt
import os 

plt.switch_backend('agg')

resolution = 0.243
level = 5

'''比较tiff and npy mask'''
#tiff_dir = '/mnt/lustre/yuxian/Code/NCRF-master/TEST_MASK/'
#mask_dir = '/mnt/lustre/yuxian/Code/NCRF-master/Data/mask_testing_cv2_level6/'
#difference = 0
#difference_tumor = 0
#num = 1
#for each in os.listdir(tiff_dir):  
#    if '.tif' in each:
#        tiff = openslide.open_slide(os.path.join(tiff_dir,each))
#        filename = each.split('.')[0]+'.npy'
#        
#        dims = tiff.level_dimensions[level]
#        pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')
#        pixelarray = np.array(tiff.read_region((0,0)*255, level, dims))[:,:,0]
#        
#        npy = np.load(os.path.join(mask_dir,filename))
#    
#        difference += 1 - float(np.count_nonzero(pixelarray==npy)/pixelarray.size)
#        print('===========The filename: %s==========='%(filename.split('.')[0]))
#
#        print('The difference: %f%%'%((1 - float(np.count_nonzero(pixelarray==npy))/pixelarray.size)*100))
#
#        tumor_region = np.max([np.count_nonzero(pixelarray),np.count_nonzero(npy)])
#        print('The total tumor region : %d'%(tumor_region))
#        intersection = np.count_nonzero(pixelarray==((npy==0)+1))
#        difference_tumor += 1 - float(intersection/tumor_region)
##        intersection = np.intersect1d(pixelarray, npy)#len(set(pixelarray==255).intersection(npy==255))
##        difference_tumor += 1 - float(intersection/tumor_region)
#        print('The difference(%d) on tumor region(%d): %f%%'%(tumor_region-intersection, tumor_region, (1 - float(intersection/tumor_region))*100))
#    
#        num += 1
#        
#print('===========The total difference============')
#difference /= num
#print('difference:%f%%'%(difference*100))
#difference_tumor /= num
#print('difference on tumor region:%f%%'%(difference_tumor*100))
#  
##dims = slide.level_dimensions[level]
##pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')
##pixelarray = np.array(slide.read_region((0,0), level, dims).convert('RGB'))    
#



'''可视化npy2jpeg看看mask'''   
 
#Prob_map_dir = '/mnt/lustre/yuxian/Code/NCRF-master/PROBS_MAP_PATH/crf/baidu_model_level6/'
#mask_dir = '/mnt/lustre/yuxian/Code/NCRF-master/Data/mask_testing_cv2_level6/' 
#save_dir = '/mnt/lustre/yuxian/Code/NCRF-master/PROBS_MAP_PATH/crf/baidu_model_level6/vis/'
#
#for each in os.listdir(Prob_map_dir):
#    if '.npy' in each:
#        filename = each[:-4]
#        npy_prob = np.load(Prob_map_dir+each, mmap_mode='r')
#        print('filename: ', filename)
#        print(np.shape(npy_prob))
#        print(np.max(npy_prob))
#        plt.imshow(npy_prob.transpose(), vmin=0, vmax=1, cmap='jet')    
#        plt.savefig(save_dir+filename+'probmap_jet.png')
#        
#        if os.path.exists(mask_dir+each):
#            npy_mask = np.load(mask_dir+each, mmap_mode='r')/255
#            print(np.shape(npy_mask))
#            print(np.max(npy_mask))
#            plt.imshow(npy_mask, vmin=0, vmax=1, cmap='jet')    
#            plt.savefig(save_dir+filename+'mask_jet.png')
#        
    
    

#npy_prob = np.load('/mnt/lustre/yuxian/Code/NCRF-master/Data/mask_testing_cv2_level6/test_084.npy', mmap_mode='r')
#print(np.shape(npy_prob))
#print(np.max(npy_prob))
#plt.imshow(npy_prob, vmin=0, vmax=1, cmap='jet')
#plt.savefig('rawnpy_convert_img_084_jet.png')



#slide = openslide.open_slide('/mnt/lustre/yuxian/Code/NCRF-master/TEST_MASK/test_084.tif')
#dims = slide.level_dimensions[level]
#pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')
#pixelarray = np.array(slide.read_region((0,0), level, dims))[:,:,0]#.convert('RGB'))    
#print("np max of raw tif:",np.max(pixelarray))
#print("np shape of raw tif", np.shape(pixelarray))
#tif_convert_img= Image.fromarray(pixelarray.astype(np.uint8))
#tif_convert_img.save('rawtiff_convert_img_084.jpeg')
#plt.imshow(tif_convert_img, vmin=0, vmax=1, cmap='jet')
#plt.savefig('rawtiff_convert_img_084_jet.png')


'''原图'''
#slide = openslide.open_slide('/mnt/lustre/share/CAMELYON16/testing/images/test_084.tif')
#dims = slide.level_dimensions[level]
#pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')
#pixelarray = np.array(slide.read_region((0,0), level, dims).convert('RGB'))    
#print("np max of raw tif:",np.max(pixelarray))
#print("np shape of raw tif", np.shape(pixelarray))
#tif_convert_img= Image.fromarray(pixelarray.astype(np.uint8))
#tif_convert_img.save('rawtiff_convert_img_084.jpeg')
  
'''Recommend plot'''
#In [1]: import numpy as np
#
#In [2]: from matplotlib import pyplot as plt
#
#In [3]: probs_map = np.load('./Test_001.npy')
#
#In [4]: plt.imshow(probs_map.transpose(), vmin=0, vmax=1, cmap='jet')
#Out[4]: <matplotlib.image.AxesImage at 0x7f35e994ca58>
#
#In [5]: plt.show()


#npy_prob = np.load('/mnt/lustre/yuxian/Code/NCRF-master/PROBS_MAP_PATH/crf/xiaodi_level6/test_043.tif-3-pmap.npy', mmap_mode='r')
#print(np.shape(npy_prob))
#print(np.max(npy_prob))
#plt.imshow(npy_prob, vmin=0, vmax=1, cmap='jet')
#plt.savefig('probmap_convert_img_043_plot_xiaodi_jet.png')
#npy_prob = np.load('/mnt/lustre/yuxian/Code/NCRF-master/PROBS_MAP_PATH/crf/resample_level6/test_043.npy', mmap_mode='r')
#print(np.shape(npy_prob))
#print(np.max(npy_prob))
#plt.imshow(npy_prob.transpose(), vmin=0, vmax=1, cmap='jet')
#plt.savefig('probmap_convert_img_043_plot_resample_jet.png')
#npy_prob = np.load('/mnt/lustre/yuxian/Code/NCRF-master/PROBS_MAP_PATH/crf/baidu_model_level6/test_043.npy', mmap_mode='r')
#print(np.shape(npy_prob))
#print(np.max(npy_prob))
#plt.imshow(npy_prob.transpose(), vmin=0, vmax=1, cmap='jet')
#plt.savefig('probmap_convert_img_043_plot_baidu_jet.png')


npy_prob = np.load('/mnt/lustre/yuxian/Data_t1/NCRF-master/PROBS_MAP_PATH/probs_test_026.npy', mmap_mode='r')
plt.imshow(npy_prob.transpose(), vmin=0, vmax=1, cmap='jet')
plt.savefig('probmap_convert_img_026_plot_zqreproduce_jet.png')

#npy_prob = np.load('/mnt/lustre/yuxian/Code/NCRF-master/PROBS_MAP_PATH/crf/retrain/test_026.npy', mmap_mode='r')
#plt.imshow(npy_prob.transpose(), vmin=0, vmax=1, cmap='jet')
#plt.savefig('probmap_convert_img_026_plot_retrain_black.png')
#
#npy_prob = np.load('/mnt/lustre/yuxian/Code/NCRF-master/PROBS_MAP_PATH/crf/resample_level6/test_026.npy', mmap_mode='r')
#plt.imshow(npy_prob.transpose(), vmin=0, vmax=1, cmap='jet')
#plt.savefig('probmap_convert_img_026_plot_resample_jet.png')
#
#npy_mask = np.load('/mnt/lustre/yuxian/Code/NCRF-master/PROBS_MAP_PATH/crf/LEVEL6/test_026.npy', mmap_mode='r')
#plt.imshow(npy_mask.transpose(), vmin=0, vmax=1, cmap='jet')
#plt.savefig('probmap_convert_img_026_plot_reproduce_jet.png')
#npy_mask = np.transpose(npy_mask)
#print("np max of npy prob:",np.max(npy_prob))
#print("np shape of npy prob", np.shape(npy_prob))
#print("np max of npy mask:",np.max(npy_mask))
#print("np shape of npy mask", np.shape(npy_mask))
#ndarray_convert_img= Image.fromarray((npy_mask*255.0).astype(np.uint8))
#ndarray_convert_img.save('probmap_convert_img_084_plot_black.jpeg')
##generate_heatmap(npy_prob,npy_mask,'heatmap_convert_img_026.jpeg')
#fig = plt.imshow(npy_mask)
#fig.set_cmap('hot')
#fig.axes.get_xaxis().set_visible(False)
#fig.axes.get_yaxis().set_visible(False)
#plt.savefig('probmap_convert_img_026_plot.png')

#distance = nd.distance_transform_edt(255 - npy_mask)
#Threshold = 75/(resolution * pow(2, level) * 2) # 75µm is the equivalent size of 5 tumor cells
#binary = distance < Threshold
#filled_image = nd.morphology.binary_fill_holes(binary)
#evaluation_mask = measure.label(filled_image, connectivity = 2) 
#ndarray_convert_img= Image.fromarray(evaluation_mask.astype(np.uint8))
#ndarray_convert_img.save('probmap_convert_img2.jpeg')



#slide = openslide.open_slide('/mnt/lustre/yuxian/Code/NCRF-master/TEST_MASK/test_084.tif')
#dims = slide.level_dimensions[level]
#pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')
#pixelarray = np.array(slide.read_region((0,0), level, dims))[:,:,0]
#plt.imshow(npy_mask, vmin=0, vmax=1, cmap='jet')
#plt.savefig('tif_mask_convert_img_084.png')
#print('pixelarray:',pixelarray)
#print("np max of npy mask pixelarray:",np.max(pixelarray))
#print("np min of npy mask pixelarray:",np.min(pixelarray))
#tif_convert_img= Image.fromarray((pixelarray*255.0).astype(np.uint8))
#tif_convert_img.save('tif_mask_convert_img_084.jpeg')
##
#distance = nd.distance_transform_edt(255 - pixelarray*255.0)####需要乘以255
#print('distance:',distance)
#print("np max distance",np.max(distance))
#print("np min distance",np.min(distance))
#print("np shape distance",np.shape(distance))
#Threshold = 75/(resolution * pow(2, level) * 2) # 75µm is the equivalent size of 5 tumor cells
#print("Threshold:",Threshold)
#binary = distance < Threshold
#filled_image = nd.morphology.binary_fill_holes(binary)
#evaluation_mask = measure.label(filled_image, connectivity = 2)    
#
#print("np max of npy mask:",np.max(evaluation_mask))
#print("np shape of npy mask", np.shape(evaluation_mask))
#tif_convert_img= Image.fromarray((evaluation_mask).astype(np.uint8))
#tif_convert_img.save('tif_mask_convert_img2.jpeg')