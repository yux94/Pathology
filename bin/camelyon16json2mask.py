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
from multiprocessing import Pool, Value, Lock
import time

np.random.seed(0)
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

count = Value('i', 0)
lock = Lock()

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
            
            
class json2mask(object):
    """
    Data producer that generate a square grid, e.g. 3x3, of patches and their
    corresponding labels from pre-sampled images.
    """
    def __init__(self, data_path, json_path, img_size=768):
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
        self._json_path = json_path
        self._img_size = img_size
        self._preprocess()

    def _preprocess(self):

        self._pids = list(map(lambda x: x.strip('.json'),
                              os.listdir(self._json_path)))

        self._annotations = {}
        logging.info('loading label ...')
        for pid in self._pids:
            pid_json_path = os.path.join(self._json_path, pid + '.json')
            anno = Annotation()
            anno.from_json(pid_json_path)
            self._annotations[pid] = anno

        self._coords = []
        logging.info('loading data ...')
        f = open(os.path.join(self._data_path, 'list.txt'))
        i=0
        for line in f:
            pid, x_center, y_center = line.strip('\n').split(',')[0:3]
            x_center, y_center = int(x_center), int(y_center)
#            self._coords.append((pid, x_center, y_center))
            '''modified'''
            self._coords.append((str(i), pid, x_center, y_center))
            i+=1
            
        f.close()

        self._num_image = len(self._coords)
        
        pool = Pool(processes=5)

#        for xx_idx in range(self._num_image):   
#            self.getitem(xx_idx)
        '''modified'''
        pool.map(self.getitem, self._coords)       
#        pool.map(self.mask__, self._coords)       
            
    def __len__(self):
        return self._num_image

#    def getitem(self, idx):#label 与原输入图大小一致，mask
#        pid, x_center, y_center = self._coords[idx]
    '''modified'''
#    def mask__(self, coords):#label 与原输入图大小一致，mask
#        idx, pid, x_center, y_center = coords
#
#        x_top_left = int(x_center - self._img_size / 2)
#        y_top_left = int(y_center - self._img_size / 2)
#
#        x_bottom_right = int(x_center + self._img_size / 2)
#        y_bottom_right = int(y_center + self._img_size / 2)
#        
##        num_of_point = len(self._annotations[pid])        
#        label_img = np.zeros((self._img_size, self._img_size),dtype=np.int64)#768x768 all black
#
#        '''load the corresponding json file (same pid)'''
##        logging.info('begin transfering {}th label image {} ({}, {})...'.format(idx, pid, x_center, y_center))
#
#        global lock
#        global count
#        
#        with open(os.path.join(self._json_path, pid + '.json')) as f:
#            annotations_json = json.load(f)
#            neg = annotations_json['negative']
#            num_name = np.size(neg)
#            
#            vertice_=np.zeros((1,2),dtype=np.int64)
#
#            for name_i in range(num_name):
#                if name_i==0:
#                    vertice = np.array(neg[name_i]['vertices'])
#                else:                           
#                    vertice = np.array(neg[name_i]['vertices'])
#                    
#                vertice_ = np.concatenate((vertice_,vertice))                            
#            
#            vertice_ = vertice_[1:]
#            
#            
#        for num_xy in range(len(vertice_)):
#            x,y = vertice_[num_xy]
#            if x< x_bottom_right and x> x_top_left and y> y_top_left and y< y_bottom_right:
#                real_x = int(x-x_top_left)
#                real_y = int(y-y_top_left)
#                label_img[real_x][real_y]=1
#                logging.info('vertice ({},{}) is the anno ...'.format(real_x,real_y))
#
#        save_path = self._data_path+'/mask/new/'
#        if not os.path.exists(save_path):
#            os.mkdir(save_path)
#        
#        np.save(os.path.join(save_path,idx+'.npy'), label_img)
#        
#        label_img = Image.fromarray(label_img,'L')            
#        label_img.save(os.path.join(save_path,idx+'.jpeg'),"jpeg")
#                
##        logging.info('saved {}th label image ...'.format(idx))
#
#        '''modified'''        
#    
#        with lock:
#            count.value += 1
#            if (count.value) % 100 == 0:
#                logging.info('{}, {} masks generated...'
#                             .format(time.strftime("%Y-%m-%d %H:%M:%S"),
#                                     count.value))
#              
#              
#              
    def getitem(self, coords):#label 与原输入图大小一致，mask
        idx, pid, x_center, y_center = coords
        
        save_path = self._data_path+'/mask/'
        save_path2 = self._data_path+'/npy/'        

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(save_path2):
            os.mkdir(save_path2)
        
        if not os.path.exists(os.path.join(save_path,idx+'.jpeg')):
            
            x_top_left = int(x_center - self._img_size / 2)
            y_top_left = int(y_center - self._img_size / 2)
       
            label_img = np.zeros((self._img_size, self._img_size),dtype=np.int64)#768x768 all black
    
            logging.info('begin transfering {}th label image {} ({}, {})...'.format(idx, pid, x_center, y_center))
            
            for x_idx in range(self._img_size):
                for y_idx in range(self._img_size):
                    # (x, y) is the center of each patch
                    x = x_top_left + x_idx
                    y = y_top_left + y_idx
    
                    if self._annotations[pid].inside_polygons((x, y), True):
                        label = 1
                    else:
                        label = 0
    
                    # extracted images from WSI is transposed with respect to
                    # the original WSI (x, y)
                    label_img[x_idx, y_idx] = label
                    
            logging.info('before saving {}th label image ...'.format(idx))
                    
    ##        save_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID/mask/'
    #        save_path = self._data_path+'/mask/'
    #        save_path2 = self._data_path+'/npy/'
    #        
    #        if not os.path.exists(save_path):
    #            os.mkdir(save_path)
    #        if not os.path.exists(save_path2):
    #            os.mkdir(save_path2)
    
    #        label_img_npy = label_img.reshape([np.shape(label_img)[1],np.shape(label_img)[0]])            
    #        np.save(os.path.join(save_path2,idx+'.npy'), label_img)
            
            '''Attention!'''
            label_img_npy = label_img.reshape([np.shape(label_img)[1],np.shape(label_img)[0]])
            label_img = Image.fromarray((label_img_npy*255).astype(np.uint8))
    #        label_img = label_img.transpose(Image.FLIP_TOP_BOTTOM)        
    #        label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)        
            label_img = label_img.rotate(90) 
            label_img = label_img.transpose(Image.FLIP_TOP_BOTTOM) 
            label_img.save(os.path.join(save_path,idx+'.jpeg'),"jpeg")
            
    #        label_img.save(os.path.join('/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID/mask/',pid+'.jpeg'),"jpeg")
            
    #        data = data*255
    #        new_im = Image.fromarray(data.astype(np.uint8))        
            
            
        logging.info('saved {}th label image ...'.format(idx))

        '''modified'''        
        global lock
        global count
    
        with lock:
            count.value += 1
            if (count.value) % 100 == 0:
                logging.info('{}, {} masks generated...'
                             .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                     count.value))
                                     
                                     
#        img = Image.open(os.path.join(self._data_path, '{}.png'.format(idx)))

#        #数据增扩
#        # color jitter
#        img = self._color_jitter(img)#对颜色的数据增强:图像亮度、饱和度、对比度变化
#
#        # use left_right flip
#        if np.random.rand() > 0.5:
#            img = img.transpose(Image.FLIP_LEFT_RIGHT)
#            label_img = np.fliplr(label_img)
#
#        # use rotate
#        num_rotate = np.random.randint(0, 4)
#        img = img.rotate(90 * num_rotate)
#        label_img = np.rot90(label_img, num_rotate)
#
#        # PIL image:   H x W x C
#        # torch image: C X H X W
#        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))
#
#        if self._normalize:
#            img = (img - 128.0)/128.0
#
#        # flatten the square grid
#        img_flat = np.zeros(
#            (self._grid_size, 3, self._crop_size, self._crop_size),
#            dtype=np.float32)
#        label_flat = np.zeros(self._grid_size, dtype=np.float32)

#        idx = 0
#        for x_idx in range(self._patch_per_side):
#            for y_idx in range(self._patch_per_side):
#                # center crop each patch
#                x_start = int(
#                    (x_idx + 0.5) * self._patch_size - self._crop_size / 2)
#                x_end = x_start + self._crop_size
#                y_start = int(
#                    (y_idx + 0.5) * self._patch_size - self._crop_size / 2)
#                y_end = y_start + self._crop_size
#                img_flat[idx] = img[:, x_start:x_end, y_start:y_end]
#                label_flat[idx] = label_img[x_idx, y_idx]
#
#                idx += 1
            
#        return img_flat, label_flat
#        return label_img

def main():
    logging.basicConfig(level=logging.INFO)
#    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID'#/home/likewise-open/SENSETIME/yuxian/Camelyon16/Baidu/NCRF-master/PATCHES_TUMOR_VALID'#/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID'
#    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_VALID'#/home/likewise-open/SENSETIME/yuxian/Camelyon16/Baidu/NCRF-master/PATCHES_TUMOR_VALID'#/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID'
#    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_TRAIN'#/home/likewise-open/SENSETIME/yuxian/Camelyon16/Baidu/NCRF-master/PATCHES_TUMOR_VALID'#/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID'
    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_TRAIN'#/home/likewise-open/SENSETIME/yuxian/Camelyon16/Baidu/NCRF-master/PATCHES_TUMOR_VALID'#/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID'

#    json_path = '/mnt/lustre/yuxian/Code/NCRF-master/jsons/valid'#/home/likewise-open/SENSETIME/yuxian/Camelyon16/Baidu/NCRF-master/jsons/valid'#/yuxian/Code/NCRF-master/json'
    json_path = '/mnt/lustre/yuxian/Code/NCRF-master/jsons/train'#/home/likewise-open/SENSETIME/yuxian/Camelyon16/Baidu/NCRF-master/jsons/valid'#/yuxian/Code/NCRF-master/json'
    json2mask(data_path,json_path)
#    json2mask(data_path,json_path)
    

if __name__ == '__main__':
    main()
