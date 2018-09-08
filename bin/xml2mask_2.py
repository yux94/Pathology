import openslide
import xml.etree.ElementTree as ET
import numpy as np
from skimage.draw import polygon
from multiprocessing import Pool
import cv2
import logging


def camelyon16xml2json(inxml, level):
    """
    Convert an annotation of camelyon16 xml format into a json format.

    Arguments:
        inxml: string, path to the input camelyon16 xml format
        outjson: string, path to the output json format
    """
    root = ET.parse(inxml).getroot()
    annotations_tumor = \
        root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
    annotations_0 = \
        root.findall('./Annotations/Annotation[@PartOfGroup="_0"]')
    annotations_1 = \
        root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
    annotations_2 = \
        root.findall('./Annotations/Annotation[@PartOfGroup="_2"]')
    annotations_positive = \
        annotations_tumor + annotations_0 + annotations_1
    annotations_negative = annotations_2

    json_dict = {}
    json_dict['positive'] = []
    json_dict['negative'] = []

    for annotation in annotations_positive:
        
        X = list(map(lambda x: float(x.get('X')),
                 annotation.findall('./Coordinates/Coordinate')))
#        print(X)
        X = [i/pow(2,level) for i in X]
        Y = list(map(lambda x: float(x.get('Y')),
                 annotation.findall('./Coordinates/Coordinate')))
        Y = [i/pow(2,level) for i in Y]
        vertices = np.round([X, Y]).astype(int).transpose().tolist()
        name = annotation.attrib['Name']
        json_dict['positive'].append({'name': name, 'vertices': vertices})

    for annotation in annotations_negative:
        X = list(map(lambda x: float(x.get('X')),
                 annotation.findall('./Coordinates/Coordinate')))
        X = [i/pow(2,level) for i in X]
        Y = list(map(lambda x: float(x.get('Y')),
                 annotation.findall('./Coordinates/Coordinate')))
        Y = [i/pow(2,level) for i in Y]
        vertices = np.round([X, Y]).astype(int).transpose().tolist()
        name = annotation.attrib['Name']
        json_dict['negative'].append({'name': name, 'vertices': vertices})

    return json_dict


def vertices2polygon(vertices):
    vertices = np.asarray(vertices, dtype=np.int)
    r = vertices[:, 1]
    c = vertices[:, 0]
    minr = r.min()
    minc = c.min()
    r -= minr
    c -= minc
    rr, cc = polygon(r, c)
    return rr + minr, cc + minc


def jsondict2mask(json_dict, xyshape):
    shape = np.asarray([xyshape[1], xyshape[0]], dtype=np.int)
    mask = np.zeros(shape, dtype=np.uint8)
    vertices = [np.asarray(anno_dict['vertices'], dtype=np.int) for anno_dict in json_dict['positive']]
    mask = cv2.fillPoly(mask, vertices, 255)
    vertices = [np.asarray(anno_dict['vertices'], dtype=np.int) for anno_dict in json_dict['negative']]
    if len(vertices) > 0:
        mask = cv2.fillPoly(mask, vertices, 0)
    return mask


def get_xyshape(tifffile, level):
#def get_xyshape(tifffile):
#    return openslide.OpenSlide(tifffile).dimensions
    return openslide.OpenSlide(tifffile).level_dimensions[level]

def main(name):
    logging.info('processing {} ...'.format(name))
    np.save(os.path.join(npy_dir, name), jsondict2mask(
#        camelyon16xml2json(os.path.join(xml_dir, name + '.xml')),
        camelyon16xml2json(os.path.join(xml_dir, name + '.xml'),level=6),
#        get_xyshape(os.path.join(tif_dir, name + '.tif'))
        get_xyshape(os.path.join(tif_dir, name + '.tif'),level=6)#level+1 从0开始
    ))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import os
    xml_dir = './CAMELYON16/training/lesion_annotations'
    tif_dir = './CAMELYON16/training/tumor'
    npy_dir = './mask_training_cv2_level7'
    os.makedirs(npy_dir, exist_ok=True)
    names = [os.path.splitext(xml_file)[0] for xml_file in sorted(os.listdir(xml_dir)) if xml_file.endswith('.xml')]
    Pool().map(main, names)


