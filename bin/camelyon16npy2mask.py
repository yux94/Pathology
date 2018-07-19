# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:40:43 2018

@author: SENSETIME\yuxian
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 19:01:55 2018

@author: SENSETIME\yuxian
"""

import os
import sys
import logging
import copy
import json
import numpy as np
from PIL import Image
from skimage.measure import points_in_poly
import logging

np.random.seed(0)
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


class Polygon(object):
    """
    Polygon represented as [N, 2] array of vertices
    """
    def __init__(self, name, vertices):
        """
        Initialize the polygon.

        Arguments:
            name: string, name of the polygon
            vertices: [N, 2] 2D numpy array of int
        """
        self._name = name
        self._vertices = vertices

    def __str__(self):
        return self._name

    def inside(self, coord):
        """
        Determine if a given coordinate is inside the polygon or not.

        Arguments:
            coord: 2 element tuple of int, e.g. (x, y)

        Returns:
            bool, if the coord is inside the polygon.
        """
        return points_in_poly([coord], self._vertices)[0]

    def vertices(self):

        return np.array(self._vertices)
        
class Annotation(object):
    """
    Annotation about the regions within WSI in terms of vertices of polygons.
    """
    def __init__(self):
        self._json_path = ''
        self._polygons_positive = []
        self._polygons_negative = []

    def __str__(self):
        return self._json_path

    def from_json(self, json_path):
        """
        Initialize the annotation from a json file.

        Arguments:
            json_path: string, path to the json annotation.
        """
        self._json_path = json_path
        with open(json_path) as f:
            annotations_json = json.load(f)

        for annotation in annotations_json['positive']:
            name = annotation['name']
            vertices = np.array(annotation['vertices'])
            polygon = Polygon(name, vertices)
            self._polygons_positive.append(polygon)

        for annotation in annotations_json['negative']:
            name = annotation['name']
            vertices = np.array(annotation['vertices'])
            polygon = Polygon(name, vertices)
            self._polygons_negative.append(polygon)

    def inside_polygons(self, coord, is_positive):
        """
        Determine if a given coordinate is inside the positive/negative
        polygons of the annotation.

        Arguments:
            coord: 2 element tuple of int, e.g. (x, y)
            is_positive: bool, inside positive or negative polygons.

        Returns:
            bool, if the coord is inside the positive/negative polygons of the
            annotation.
        """
        if is_positive:
            polygons = copy.deepcopy(self._polygons_positive)
        else:
            polygons = copy.deepcopy(self._polygons_negative)

        for polygon in polygons:
            if polygon.inside(coord):
                return True

        return False

    def polygon_vertices(self, is_positive):
        """
        Return the polygon represented as [N, 2] array of vertices

        Arguments:
            is_positive: bool, return positive or negative polygons.

        Returns:
            [N, 2] 2D array of int
        """
        if is_positive:
            return list(map(lambda x: x.vertices(), self._polygons_positive))
        else:
            return list(map(lambda x: x.vertices(), self._polygons_negative))
             
class npy2mask(object):
    """
    Data producer that generate a square grid, e.g. 3x3, of patches and their
    corresponding labels from pre-sampled images.
    """
    def __init__(self, data_path, npy_path, img_size=768):
        """
        Initialize the data producer.

        Arguments:
            data_path: string, path to pre-sampled images using patch_gen.py
            json_path: string, path to the annotations in json format
            img_size: int, size of pre-sampled images, e.g. 768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
        """
        self._data_path = data_path
        self._npy_path = npy_path
        self._img_size = img_size
        self._preprocess()

    def _preprocess(self):

        self._pids = list(map(lambda x: x.strip('.npy'),
                              os.listdir(self._npy_path)))

        self._coords = []
        logging.info('loading patch ...')
        f = open(self._data_path)
        index = 0
        for line in f:
            pid, x_center, y_center = line.strip('\n').split(',')[0:3]
            x_center, y_center = int(x_center), int(y_center)
            self._coords.append((pid, x_center, y_center, index))
            index += 1
            
        self._coords.sort()    
        f.close()

        self._num_patches = len(self._coords)       
       
        old_pid = 0
        wsi_mask = []
        for xx_idx in range(self._num_patches):   
            old_pid, wsi_mask = self.getitem(xx_idx, old_pid, wsi_mask)
            
    def __len__(self):
        return self._num_patches###

    def getitem(self, idx, old_pid, wsi_mask):#label 与原输入图大小一致，mask
    
#        save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID/mask_HZQ/'
#        save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_TRAIN/mask_HZQ/'
#        save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_VALID/mask_HZQ/'
        save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_TRAIN/mask_HZQ/'
                       
        pid, x_center, y_center, index = self._coords[idx]

#        x_top_left = int(x_center - self._img_size / 2)
#        y_top_left = int(y_center - self._img_size / 2)
       
           
#        if not os.path.exists(save_path+str(index)+'.jpeg') or os.path.getsize(save_path+str(index)+'.jpeg')==0:   
        if index==127328:
            x_top_left = int(x_center - self._img_size / 2) 
            y_top_left = int(y_center - self._img_size / 2)
            if x_top_left<0:
                x_top_left = 0
            if y_top_left<0:
                y_top_left = 0            
            
            label_img = np.zeros((self._img_size, self._img_size),#768x768
                                  dtype=np.float32)
            
            if 'normal' not in pid:
                if old_pid == pid and np.size(wsi_mask)!=0:
                    pass
                else:
                    wsi_mask = np.load(self._npy_path+pid+'.npy')#, mmap_mode='r')    
                    print('crop mask from wsi npy ...', np.shape(wsi_mask))
                
                print('wsi_mask shape:', np.shape(wsi_mask))
                print(y_top_left)
                print(y_top_left + self._img_size)
                print(x_top_left)
                print(x_top_left + self._img_size)
                
                label_img =  wsi_mask[y_top_left:y_top_left + self._img_size, x_top_left:x_top_left + self._img_size]#, y_top_left:y_top_left + self._img_size]               
    #            label_img =  wsi_mask[x_top_left:x_top_left + self._img_size, y_top_left:y_top_left + self._img_size]               
            
#            print(np.max(label_img))            
    #        label_img = np.transpose(label_img)    
            label_img = Image.fromarray(label_img.astype(np.uint8))  
    
            
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            
            label_img.save(save_path+str(index)+'.jpeg',"jpeg")
    
              
        logging.info('saving mask patch {}/{} ...pid: {}  index: {}'.format(idx,self._num_patches,pid, index))
    
        return pid, wsi_mask

def main():
    logging.basicConfig(level=logging.INFO)
#    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/coords/tumor_valid.txt'
#    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/coords/tumor_train.txt'
#    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/coords/normal_valid.txt'
    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/coords/normal_train.txt'
    npy_path = '/mnt/lustre/share/CAMELYON16/masks_training_cv2/'

    npy2mask(data_path,npy_path)
    

if __name__ == '__main__':
    main()
