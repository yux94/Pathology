# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 09:28:03 2018

@author: SENSETIME\yuxian
"""

#!/usr/bin/env python
# coding: utf-8
 
def getfilelines(filename, eol='\n', buffsize=4096):
    """计算给定文件有多少行"""
    with open(filename, 'rb') as handle:
        linenum = 0
        buffer = handle.read(buffsize)
        while buffer:
            linenum += buffer.count(eol)
            buffer = handle.read(buffsize)
        return linenum
 
 
def readtline(filename, lineno, eol="\n", buffsize=4096):
    """读取文件的指定行"""
    with open(filename, 'rb') as handle:
        readedlines = 0
        buffer = handle.read(buffsize)
        while buffer:
            thisblock = buffer.count(eol)
            if readedlines < lineno < readedlines + thisblock:
                # inthisblock: findthe line content, and return it
                return buffer.split(eol)[lineno - readedlines - 1]
            elif lineno == readedlines + thisblock:
                # need continue read line rest part
                part0 = buffer.split(eol)[-1]
                buffer = handle.read(buffsize)
                part1 = buffer.split(eol)[0]
                return part0 + part1
            readedlines += thisblock
            buffer = handle.read(buffsize)
        else:
            raise IndexError
 
 
def getrandomline(filename):
    """读取文件的任意一行"""
    import random
    return readtline(
        filename,
        random.randint(0, getfilelines(filename)),
        )
 
 
if __name__ == "__main__":
    import sys
    import os
    import random
#    if len(sys.argv) == 1:
#        print getrandomline("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_jpeg_mask/Resample_negative.txt")
##        print getrandomline("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_jpeg_mask/Resample_positive.txt")
#    
#    else:
#        for f in filter(os.path.isfile, sys.argv[1:]):
#            print getrandomline(f)
    
    
#    negative_train = open("/mnt/lustre/yuxian/Code/NCRF-master/coords/resample/normal_train.txt",'w')
#    negative_valid = open("/mnt/lustre/yuxian/Code/NCRF-master/coords/resample/normal_valid.txt",'w')
    positive_train = open("/mnt/lustre/yuxian/Code/NCRF-master/coords/resample/tumor_train.txt",'w')
    positive_valid = open("/mnt/lustre/yuxian/Code/NCRF-master/coords/resample/tumor_valid.txt",'w')
    lines=[]
    
#    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_jpeg_mask/Resample_negative.txt", 'r') as infile:
    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_jpeg_mask/Resample_positive.txt", 'r') as infile:
        
        for line in infile:
            lines.append(line)
    
    random.shuffle(lines)
    
    print('length of txt:', len(lines))
#    NUM = 0
    for idx in range(len(lines)):
        if idx>=200000:
            break
        else:

            pid, x_center, y_center = lines[idx].strip('\n').split(',')
#
#            if "tumor_111" or "normal_144" in pid:
#                pass
#            else:
#            print()
            positive_train.write(lines[idx])
#            negative_train.write(lines[idx])
            
            if idx<20000:
                positive_valid.write(lines[len(lines)-idx-1])
#                NUM += 1
#                negative_valid.write(lines[len(lines)-idx-1])
            
    infile.close()
    positive_train.close()
    positive_valid.close()
#    negative_train.close()
#    negative_valid.close()
        
