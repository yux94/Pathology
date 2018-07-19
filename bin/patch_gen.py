import sys
import os
import argparse
import logging
import time
from shutil import copyfile
from multiprocessing import Pool, Value, Lock

import openslide
   
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

#srun --partition=MIA python ./wsi/bin/patch_gen.py ./Data/WSI_TRAIN/ ./coords/tumor_train.txt ./Data/PATCHES_TUMOR_TRAIN/ 
#python ./wsi/bin/patch_gen.py ./Data/WSI_TRAIN/ /coords/tumor_train.txt ./Data/PATCHES_TUMOR_TRAIN/
#python ./wsi/bin/patch_gen.py ./Data/WSI_TRAIN/ ./coords/normal_train1.txt ./Data/PATCHES_NORMAL_TRAIN/1/
#python /wsi/bin/patch_gen.py /Data/WSI_VAL/ /coords/tumor_valid.txt /Data/PATCHES_TUMOR_VALID/
#python ./wsi/bin/patch_gen.py ./Data/WSI_TRAIN/ ./coords/normal_valid.txt ./Data/PATCHES_NORMAL_VALID/

parser = argparse.ArgumentParser(description='Generate patches from a given '
                                 'list of coordinates')
parser.add_argument('wsi_path', default='/Data/WSI_TRAIN_VAL/', metavar='WSI_PATH', type=str,
#parser.add_argument('wsi_path', default='/Data/WSI_TRAIN/', metavar='WSI_PATH', type=str,
                    help='Path to the input directory of WSI files')
parser.add_argument('coords_path', default='/coords/resample/normal_train.txt', metavar='COORDS_PATH',
#parser.add_argument('coords_path', default='/coords/normal_train.txt', metavar='COORDS_PATH',
                    type=str, help='Path to the input list of coordinates')#normal_train   normal_valid   tumor_train   tumor_valid
parser.add_argument('patch_path', default='/Data/resample/PATCHES_NORMAL_TRAIN/', metavar='PATCH_PATH', type=str,
#parser.add_argument('patch_path', default='/Data/PATCHES_NORMAL_TRAIN/', metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')#PATCHES_NORMAL_TRAIN  PATCHES_NORMAL_VALID  PATCHES_TUMOR_TRAIN  PATCHES_TUMOR_VALID
parser.add_argument('--patch_size', default=768, type=int, help='patch size, '
                    'default 768')
parser.add_argument('--level', default=0, type=int, help='level for WSI, to '
                    'generate patches, default 0')
parser.add_argument('--num_process', default=5, type=int,
                    help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()


def process(opts):
#def process(i, pid, x_center, y_center, args):
    i, pid, x_center, y_center, args = opts
    x = int(int(x_center) - args.patch_size / 2)
    y = int(int(y_center) - args.patch_size / 2)
    wsi_path = os.path.join(args.wsi_path, pid + '.tif')
    try:
        slide = openslide.OpenSlide(wsi_path)
    #    slide = openslide.open_slide(wsi_path)
        
#        if not (pid=='normal_031'): 
        img = slide.read_region(
            (x, y), args.level,
            (args.patch_size, args.patch_size)).convert('RGB')
        
        img.save(os.path.join(args.patch_path, str(i) + '.jpeg'))
            
    except:
        logging.info('{} image wrong...'
                     .format((i, pid, x_center, y_center)))
#        raise

#    img.save(os.path.join(args.patch_path, str(i) + '.jpeg'))
    
#    logging.info('{}, {} th image generated...'
#                         .format((x_center, y_center),
#                                 pid))
    global lock
    global count
   
    with lock:
        count.value += 1
        if (count.value) % 100 == 0:
            logging.info('{}, {} patches generated...'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                 count.value))


def run(args):
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.patch_path):
        os.mkdir(args.patch_path)

    copyfile(args.coords_path, os.path.join(args.patch_path, 'list.txt'))

    opts_list = []
    infile = open(args.coords_path)
    for i, line in enumerate(infile):
        pid, x_center, y_center = line.strip('\n').split(',')
        opts_list.append((i, pid, int(float(x_center)), int(float(y_center)), args))
#        process(i, pid, x_center, y_center, args)
        
    infile.close()
    
    pool = Pool(processes=args.num_process)
    pool.map(process, opts_list)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
