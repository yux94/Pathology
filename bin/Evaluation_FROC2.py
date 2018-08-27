# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:23:22 2018

@author: SENSETIME\yuxian
"""

'''
Created on 5 Jan 2017

@author: hjlin
'''
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:09:32 2016

@author: Babak Ehteshami Bejnordi

Evaluation code for the Camelyon16 challenge on cancer metastases detecion
"""

import openslide
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
import os
import sys
from matplotlib.ticker import MultipleLocator
plt.switch_backend('agg')

   
def computeEvaluationMask(maskDIR, resolution, level):
    """Computes the evaluation mask.
    
    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made
        
    Returns:
        evaluation_mask
    """
#    if os.path.exists( maskDIR ) == True:
    slide = openslide.open_slide(maskDIR)
    dims = slide.level_dimensions[level]
    pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')
    pixelarray = np.array(slide.read_region((0,0), level, dims))
    print(np.max(pixelarray[:,:,0]))
    if np.max(pixelarray[:,:,0])==1:
        pixelarray *= 255
    print(np.max(pixelarray[:,:,0]))
    distance = nd.distance_transform_edt(255 - pixelarray[:,:,0])
    Threshold = 75/(resolution * pow(2, level) * 2) 
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity = 2) 
#    else:
#        evaluation_mask = np.zeros( (10000,10000), dtype='int64')
    return evaluation_mask
    
    
def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)
    
    Description:
        

        
    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made
        
    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)    
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = [] 
    threshold = 275/(resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i+1)
    return Isolated_Tumor_Cells


def readCSVContent(csvDIR):
    """Reads the data inside CSV file
    
    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image
        
    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
    """
    Xcorr, Ycorr, Probs = ([] for i in range(3))
    csv_lines = open(csvDIR,"r").readlines()
    for i in range(len(csv_lines)):
        line = csv_lines[i]
        elems = line.rstrip().split(',')
        Probs.append(float(elems[0]))
        Xcorr.append(int(elems[1]))
        Ycorr.append(int(elems[2]))
    return Probs, Xcorr, Ycorr
    
         
def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells, level):
    """Generates true positive and false positive stats for the analyzed image
    
    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
        level:      The level at which the evaluation mask was made
         
    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections
        
        TP_probs:   A list containing the probabilities of the True positive detections
        
        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)
        
        detection_summary:   A python dictionary object with keys that are the labels 
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate]. 
        Lesions that are missed by the algorithm have an empty value.
        
        FP_summary:   A python dictionary object with keys that represent the 
        false positive finding number and values that contain detection 
        details [confidence score, X-coordinate, Y-coordinate]. 
    """

    max_label = np.amax(evaluation_mask)
    FP_probs = [] 
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}  
    FP_summary = {}
    for i in range(1,max_label+1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = []        
     
    FP_counter = 0       
    if (is_tumor):
        for i in range(0,len(Xcorr)):
#            HittedLabel = evaluation_mask[Ycorr[i]/pow(2, level), Xcorr[i]/pow(2, level)]
            HittedLabel = evaluation_mask[Xcorr[i]/pow(2, level), Ycorr[i]/pow(2, level)]
            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                key = 'FP ' + str(FP_counter)
                FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                FP_counter+=1
            elif HittedLabel not in Isolated_Tumor_Cells:
                if (Probs[i]>TP_probs[HittedLabel-1]):
                    label = 'Label ' + str(HittedLabel)
                    detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                    TP_probs[HittedLabel-1] = Probs[i]                                     
    else:
        for i in range(0,len(Xcorr)):
            FP_probs.append(Probs[i]) 
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]] 
            FP_counter+=1
            
    num_of_tumors = max_label - len(Isolated_Tumor_Cells);                             
    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary
 
 
def computeFROC(FROC_data):
    """Generates the data required for plotting the FROC curve
    
    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image
         
    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """
    
    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist] 
    
    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())    
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs)/float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs)/float(sum(FROC_data[3]))      
    return  total_FPs, total_sensitivity
   
   
def plotFROC(total_FPs, total_sensitivity, figPath):
    """Plots the FROC curve
    
    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
         
    Returns:
        -
    """    
    fig = plt.figure()
    plt.xlabel('Average Number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)  
    fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)
    plt.plot(total_FPs, total_sensitivity, '-', color='#000080')
    plt.xlim(0,8)
    plt.ylim(0,1)
    plt.grid(True)
    
    ax = plt.subplot(111)
    xmajorLocator = MultipleLocator(1)
    xminorLocator = MultipleLocator(0.25)
    ymajorLocator = MultipleLocator(0.1)
    yminorLocator = MultipleLocator(0.05)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.xaxis.grid(True, which='minor')
    ax.yaxis.grid(True, which='major')
    
    
    plt.savefig(figPath)    
    plt.show()       

def findPosition(total_FPs, value):      
    diff = 1000000.0
    Position_ID = -1
    for i in range(len(total_FPs)):
        t_diff = abs(total_FPs[i]-value)
        if t_diff<diff:
            Position_ID = i
            diff = t_diff
    return Position_ID
    
if __name__ == "__main__":

    mask_folder = "/mnt/lustrenew/yuxian/Code/NCRF-master/WSI_PATH/"
    result_folder = '/mnt/lustre/yuxian/Data_t1/NCRF-master/COORD_PATH/crf/xiaodi_1st_retestwithtrans_thr/'
#    result_folder = "/media/CUDisk1/hjlin/Backup/Camelyon2016/Program/Camelyon_FCN/TestTrainData/FCNTestRunning/Submit3/FinalResults/Locations_Median"
    figName = "Figure.png"
    txtName = "Figure.txt"
    EVALUATION_MASK_LEVEL = 5
    
#    mask_folder = sys.argv[1]
#    result_folder = sys.argv[2]
#    figName = sys.argv[3]
#    txtName = sys.argv[4]
#    EVALUATION_MASK_LEVEL = int(sys.argv[5])
    
    result_file_list = []
    result_file_list += [each for each in os.listdir(result_folder) if each.endswith('.csv')]
    
#     EVALUATION_MASK_LEVEL = 5 # Image level at which the evaluation is done
    L0_RESOLUTION = 0.243 # pixel resolution at level 0
    
    FROC_data = np.zeros((4, len(result_file_list)), dtype=np.object)
    FP_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    detection_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    
 
    ground_truth_test = []
    ground_truth_test += [each[0:8] for each in os.listdir(mask_folder) if each.endswith('.tif')]
    ground_truth_test = set(ground_truth_test)

    caseNum = 0    
    for case in result_file_list:
        print( 'Evaluating Performance on image:', case[0:8])
        sys.stdout.flush()
        csvDIR = os.path.join(result_folder, case)
        Probs, Xcorr, Ycorr = readCSVContent(csvDIR)
                
        is_tumor = case[0:8] in ground_truth_test  
        if (is_tumor):
            maskDIR = os.path.join(mask_folder, case[0:8]) + '.tif'
            evaluation_mask = computeEvaluationMask(maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            ITC_labels = computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
        else:
            evaluation_mask = 0
            ITC_labels = []
            
           
        FROC_data[0][caseNum] = case[0:8]
        FP_summary[0][caseNum] = case[0:8]
        detection_summary[0][caseNum] = case[0:8]
        FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], FP_summary[1][caseNum] = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels, EVALUATION_MASK_LEVEL)
        caseNum += 1
    
    # Compute FROC curve 
    total_FPs, total_sensitivity = computeFROC(FROC_data)
    sum = 0.0
    #calculate 1/4, 1/2, 1, 2, 4, 8
    file = open(result_folder+ "/" + txtName, 'w')
    po = findPosition(total_FPs, 0.25)
    file.write("1/4 :( " + str(total_FPs[po]) + ", " + str(total_sensitivity[po]) + " )\n" )
    sum = sum + total_sensitivity[po]
    po = findPosition(total_FPs, 0.5)
    file.write("1/2 :( " + str(total_FPs[po]) + ", " + str(total_sensitivity[po]) + " )\n" )
    sum = sum + total_sensitivity[po]
    po = findPosition(total_FPs, 1.0)
    file.write("1 :( " + str(total_FPs[po]) + ", " + str(total_sensitivity[po]) + " )\n" )
    sum = sum + total_sensitivity[po]
    po = findPosition(total_FPs, 2.0)
    file.write("2 :( " + str(total_FPs[po]) + ", " + str(total_sensitivity[po]) + " )\n" )
    sum = sum + total_sensitivity[po]
    po = findPosition(total_FPs, 4.0)
    file.write("4 :( " + str(total_FPs[po]) + ", " + str(total_sensitivity[po]) + " )\n" )
    sum = sum + total_sensitivity[po]
    po = findPosition(total_FPs, 8.0)
    file.write("8 :( " + str(total_FPs[po]) + ", " + str(total_sensitivity[po]) + " )\n" )
    sum = sum + total_sensitivity[po]
    file.write("Average Sensitive : " + str(sum/6.0) + " \n" )
    file.close()
    
    # plot FROC curve
    figPath = result_folder + "/" +figName
    plotFROC(total_FPs, total_sensitivity, figPath)

  
            
        
        
        
        
        
        