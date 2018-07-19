# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 14:24:32 2018

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
import openslide
import scipy.ndimage

np.random.seed(0)
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

count = Value('i', 0)
lock = Lock()

def filter_vertices(vertices, y_top_left, x_top_left, img_size):
    
    ycorners=vertices[:,1]-y_top_left
    xcorners=vertices[:,0]-x_top_left
    
    idx_y = list(np.where(ycorners>=0)[0])
    ycorners = ycorners[idx_y]
    xcorners = xcorners[idx_y]
    idx_y = list(np.where(ycorners<=0+img_size)[0])
    ycorners = ycorners[idx_y]
    xcorners = xcorners[idx_y]

    idx_x = list(np.where(xcorners>=0)[0])
    xcorners = xcorners[idx_x]
    ycorners = ycorners[idx_x]
    idx_x = list(np.where(xcorners<=0+img_size)[0])
    xcorners = xcorners[idx_x]
    ycorners = ycorners[idx_x]
    
    filtered_ver = np.transpose(np.vstack((xcorners,ycorners)))
    
    return filtered_ver

    
def create_wsi_mask(wsi, tissue_mask, img_size, annotations, wsi_label_img):#(wsi, tissue_mask, self._img_size, self._annotations[pid])

    _X_idcs, _Y_idcs = np.where(tissue_mask)
    _idcs_num = len(_X_idcs) 

    logging.info('length of tissue mask {}...'.format(_idcs_num))

    '''tissue mask patch'''
    for idx in range(_idcs_num):
        logging.info('{}th tissue mask...'.format(idx))
        
        x_mask, y_mask = _X_idcs[idx], _Y_idcs[idx]

        x_center = int((x_mask + 0.5) )
        y_center = int((y_mask + 0.5) )

        x_top_left = int(x_center - img_size / 2)
        y_top_left = int(y_center - img_size / 2)#左上角

#        patch = wsi.read_region(
#            (x_top_left, y_top_left), 0, (img_size, img_size)).convert('RGB')     #tissue region in wsi 
        
        label_img = np.zeros((img_size, img_size),dtype=np.int64)#768x768 all black
             
        '''annotations.inside_polygons((x, y), True)'''        
        for x_idx in range(img_size):
            for y_idx in range(img_size):
#                logging.info('create patch mask ({},{})...'.format(x_idx,y_idx))
                # (x, y) is the center of each patch
                x = x_top_left + x_idx
                y = y_top_left + y_idx

                if annotations.inside_polygons((x, y), True):
                    label = 1
                else:
                    label = 0

                label_img[x_idx, y_idx] = label      
                
        wsi_label_img[x_mask, y_mask] = label_img
   
    return wsi_label_img     

def create_polygon_mask(img_size, level, annotations):
    '''
    Give image and x/y coners to create a polygon mask    
    image: 2d array
    xcorners, list, points of x coners
    ycorners, list, points of y coners
    Return:
    the polygon mask: 2d array, the polygon pixels with values 1 and others with 0  
    '''
    from skimage.draw import polygon  
    
    num_group = len(annotations._polygons_positive)   ###positive : tumor polygon
    vertices = annotations._polygons_positive[0].vertices()/(pow(2, level-1))
    
#    vertices = filter_vertices(vertices, y_top_left, x_top_left, img_size)
    for group_i in range(1,num_group):
        new_ver = annotations._polygons_positive[group_i].vertices()/(pow(2, level-1))
#        new_ver = filter_vertices(new_ver, y_top_left, x_top_left, img_size)
        vertices = np.concatenate((vertices, new_ver), axis=0)
#        logging.info('concate vertices ...')
    pass
#    
#    ycorners=vertices[:,1]-y_top_left
#    xcorners=vertices[:,0]-x_top_left
#    
#    idx_y = list(np.where(ycorners>=0)[0])
#    ycorners = ycorners[idx_y,:]
#    xcorners = xcorners[idx_y,:]
#    idx_y = list(np.where(ycorners<=0+img_size)[0])
#    ycorners = ycorners[idx_y,:]
#    xcorners = xcorners[idx_y,:]
#
#    idx_x = list(np.where(xcorners>=0)[0])
#    xcorners = xcorners[idx_x,:]
#    ycorners = ycorners[idx_x,:]
#    idx_x = list(np.where(xcorners<=0+img_size)[0])
#    xcorners = xcorners[idx_x,:]
#    ycorners = ycorners[idx_x,:]

#    bst_mask = np.zeros([img_size,img_size] , dtype = np.uint8) 
    bst_mask = np.zeros(img_size, dtype = np.uint8) 

#    
    ycorners=list(vertices[:,1])
    xcorners=list(vertices[:,0])
#    logging.info('polygon vertices ...')
    if np.size(xcorners):
        rr, cc = polygon(ycorners,xcorners)       
        bst_mask[rr,cc] =1            
        
    #full_mask= ~bst_mask    
    return bst_mask 
    
    
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
            name = annotation['name']#[
            vertices = np.array(annotation['vertices'])#[None,2]    (x,y)
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
    def __init__(self, data_path, json_path, wsi_path, tissue_mask_path, img_size=768):
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
        self._wsi_path = wsi_path
        self._tissue_mask_path = tissue_mask_path
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
            '''modified'''
            self._coords.append((str(i), pid, x_center, y_center))
            i+=1
            
        f.close()

        self._num_image = len(self._coords)
        
        pool = Pool(processes=5)

        '''modified'''
        pool.map(self.getitem, self._coords)       
            
    def __len__(self):
        return self._num_image
             
    def getitem(self, coords):#label 与原输入图大小一致，mask
        idx, pid, x_center, y_center = coords# patch 中心坐标
        
        save_path = self._data_path+'/mask/tissue/'
        save_path2 = self._wsi_path+'/mask/tissue/'        

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(save_path2):
            os.mkdir(save_path2)
        
        if not os.path.exists(os.path.join(save_path,idx+'.jpeg')):
            
            x_top_left = int(x_center - self._img_size / 2)
            y_top_left = int(y_center - self._img_size / 2)
            
            wsi_path = os.path.join(self._wsi_path, pid.lower() + '.tif')
            tissue_mask_path = os.path.join(self._tissue_mask_path, pid.lower() + '.npy')
#            logging.info('wsi file path {}...'.format(wsi_path))
            slide = openslide.open_slide(wsi_path)            
            shape_level0 = slide.dimensions   #level 0 size 
               
            tissue_mask = np.load(tissue_mask_path)

            X_slide, Y_slide = slide.level_dimensions[0]
            X_mask, Y_mask = tissue_mask.shape
            if X_slide / X_mask != Y_slide / Y_mask:
                raise Exception('Slide/Mask dimension does not match ,'
                            ' X_slide / X_mask : {} / {},'
                            ' Y_slide / Y_mask : {} / {}'
                            .format(X_slide, X_mask, Y_slide, Y_mask))
            
            if not os.path.exists(os.path.join(save_path2, pid.lower() +'.npy')):
                logging.info('create wsi mask {}...'.format(wsi_path))
                
                mask = np.zeros((X_slide, Y_slide),dtype=np.int64)#768x768 all black
                
                mask = create_wsi_mask(slide, tissue_mask, self._img_size, self._annotations[pid], mask)#768
            
                np.save(os.path.join(save_path2, pid.lower()+'.npy'), mask)#wsi mask
                
            else:
                mask = np.load(os.path.join(save_path2, pid.lower() +'.npy'))

                    
            logging.info('high level mask image size...{}'.format(str(np.shape(mask))))
            mask = np.reshape(mask, shape_level0)        
            '''Attention!'''
#            label_img_npy = mask.reshape([np.shape(mask)[1],np.shape(mask)[0]])
#            label_img = Image.fromarray((label_img_npy*255).astype(np.uint8))
#            label_img = Image.fromarray((mask*255).astype(np.uint8))
#            label_img = scipy.ndimage.zoom(mask, (pow(2, level-1)), order=0)
#            logging.info('level 0 mask image size...{}'.format(str(np.shape(label_img))))
            
#            label_img = label_img.rotate(90) 
#            label_img = label_img.transpose(Image.FLIP_TOP_BOTTOM) 
            
#            label_patch =  label_img.crop((x_top_left,y_top_left,x_top_left+768,y_top_left+768))           
##            label_patch =  label_img.crop((x_top_left/(pow(2, level-1)),y_top_left/(pow(2, level-1)),int((x_top_left+768)/(pow(2, level-1))),int((y_top_left+768)/(pow(2, level-1)))))           
#            logging.info('high level mask image size...{}'.format(str(np.shape(label_patch))))
            
#            idx = 768/(np.shape(label_patch)[0])
#            label_patch = scipy.ndimage.zoom(label_patch, idx, order=0)
            label_patch = Image.fromarray((mask*255).astype(np.uint8))
#            label_img.save(os.path.join(save_path,idx+'.jpeg'),"jpeg")     
            logging.info('saving label image ...{}'.format(save_path+str(idx)+'.jpeg'))
            
            label_patch.save(os.path.join(save_path, idx+'.jpeg'),"jpeg")     
            
            
        '''modified'''        
        global lock
        global count
    
        with lock:
            count.value += 1
            if (count.value) % 100 == 0:
                logging.info('{}, {}/{} masks generated ...'
                             .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                     count.value, self._num_image))
#                                     

def main():
    logging.basicConfig(level=logging.INFO)
    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID'#/home/likewise-open/SENSETIME/yuxian/Camelyon16/Baidu/NCRF-master/PATCHES_TUMOR_VALID'#/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID'
#    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_VALID'#/home/likewise-open/SENSETIME/yuxian/Camelyon16/Baidu/NCRF-master/PATCHES_TUMOR_VALID'#/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID'
#    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_TRAIN'#/home/likewise-open/SENSETIME/yuxian/Camelyon16/Baidu/NCRF-master/PATCHES_TUMOR_VALID'#/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID'
#    data_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/PATCHES_NORMAL_TRAIN'#/home/likewise-open/SENSETIME/yuxian/Camelyon16/Baidu/NCRF-master/PATCHES_TUMOR_VALID'#/yuxian/Code/NCRF-master/Data/PATCHES_TUMOR_VALID'
    wsi_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/WSI_VAL'
#    wsi_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/WSI_TRAIN'
    json_path = '/mnt/lustre/yuxian/Code/NCRF-master/jsons/valid'#/home/likewise-open/SENSETIME/yuxian/Camelyon16/Baidu/NCRF-master/jsons/valid'#/yuxian/Code/NCRF-master/json'
#    json_path = '/mnt/lustre/yuxian/Code/NCRF-master/jsons/train'#/home/likewise-open/SENSETIME/yuxian/Camelyon16/Baidu/NCRF-master/jsons/valid'#/yuxian/Code/NCRF-master/json'
    tissue_mask_path = '/mnt/lustre/yuxian/Code/NCRF-master/Data/Tissue_mask'
    json2mask(data_path,json_path,wsi_path, tissue_mask_path)
#    json2mask(data_path,json_path)
    

if __name__ == '__main__':
    main()
