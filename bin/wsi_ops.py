import cv2
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError


NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX = 100
NUM_POSITIVE_PATCHES_FROM_EACH_BBOX = 500
PATCH_INDEX_NEGATIVE = 200000#700000
PATCH_INDEX_POSITIVE = 200000

TUMOR_PROB_THRESHOLD = 0.90
PIXEL_WHITE = 255
PIXEL_BLACK = 0
PATCH_SIZE = 768#256


def write_to_txt(txt_list,path):
#Resample_negative.txt 
#Resample_positive.txt 
    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_jpeg_mask/"+path,"a+") as f:
#    with open("/mnt/lustre/yuxian/Code/NCRF-master/Data/txt_jpeg_mask/Resample_positive.txt","a+") as f:

#        for line_idx in range(len(txt_list)):     
    
        f.write(txt_list+'\n')   
            
#        print('writing to txt with line:',line_idx)
        
        f.close()  
        
        
class PatchExtractor(object):
    @staticmethod
    def extract_positive_patches_from_tumor_region(wsi_image, tumor_gt_mask, level_used,
                                                   bounding_boxes, patch_save_dir, patch_prefix,
                                                   patch_index, wsi_name):
        """

            Extract positive patches targeting annotated tumor region

            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param tumor_gt_mask:
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to tumor regions
            :param patch_save_dir: directory to save patches into
            :param patch_prefix: prefix for patch name
            :param patch_index:
            :return:
        """

        mag_factor = pow(2, level_used)
#        tumor_gt_mask = cv2.cvtColor(tumor_gt_mask, cv2.COLOR_BGR2GRAY)
        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))
        bounding_box_num = 0
        for bounding_box in bounding_boxes:
            bounding_box_num += 1           
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            X = np.random.random_integers(b_x_start, high=b_x_end, size=NUM_POSITIVE_PATCHES_FROM_EACH_BBOX)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=NUM_POSITIVE_PATCHES_FROM_EACH_BBOX)

#            txt_list_total = []
            for x, y in zip(X, Y):
                if int(tumor_gt_mask[y-1, x-1]) is PIXEL_WHITE:#tumor
#                    print('coord (%d,%d) with %d th bounding_box'%(x,y,bounding_box_num))
#                    patch = wsi_image.read_region((x * mag_factor, y * mag_factor), 0,
#                                                  (PATCH_SIZE, PATCH_SIZE))
#                    patch.save(patch_save_dir + patch_prefix + str(patch_index), 'jpeg')
#                    patch.save(patch_save_dir + str(patch_index), 'jpeg')
                    txt_list = wsi_name+','+ str((x-1) * mag_factor + PATCH_SIZE/2) + ',' + str((y-1) * mag_factor + PATCH_SIZE/2)
#                    write_to_txt(txt_list, "Resample_tumor_valid.txt")
                    write_to_txt(txt_list, "Resample_positive.txt")
#                    txt_list_total.append(txt_list)
                    patch_index += 1
#                    patch.close()

        return patch_index

    @staticmethod
    def extract_negative_patches_from_normal_wsi(wsi_image, image_open, level_used,
                                                 bounding_boxes, patch_save_dir, patch_prefix,
                                                 patch_index, wsi_name):
        """
            Extract negative patches from Normal WSIs

            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param image_open:
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to detected ROIs
            :param patch_save_dir: directory to save patches into
            :param patch_prefix: prefix for patch name
            :param patch_index:
            :return:

        """

        mag_factor = pow(2, level_used)

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))
        bounding_box_num = 0
        for bounding_box in bounding_boxes:
            bounding_box_num += 1           
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            X = np.random.random_integers(b_x_start, high=b_x_end, size=NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)

#            txt_list_total = []
            for x, y in zip(X, Y):
                if int(image_open[y-1, x-1]) is not PIXEL_BLACK:#tissue 
#                    print('coord (%d,%d) with %d th bounding_box'%(x,y,bounding_box_num))
#                    patch = wsi_image.read_region((x * mag_factor, y * mag_factor), 0,
#                                                  (PATCH_SIZE, PATCH_SIZE))
#                    patch.save(patch_save_dir + patch_prefix + str(patch_index), 'PNG')
#                    patch.save(patch_save_dir + str(patch_index), 'jpeg')
                    txt_list = wsi_name+','+ str((x-1) * mag_factor + PATCH_SIZE/2) + ',' + str((y-1) * mag_factor + PATCH_SIZE/2)
#                    write_to_txt(txt_list, "Resample_normal_valid.txt")
                    write_to_txt(txt_list, "Resample_negative.txt")
                    patch_index += 1
#                    txt_list_total.append(txt_list)
#                    patch.close()

        return patch_index#, txt_list_total

    @staticmethod
    def extract_negative_patches_from_tumor_wsi(wsi_image, tumor_gt_mask, image_open, level_used,
                                                bounding_boxes, patch_save_dir, patch_prefix,
                                                patch_index, wsi_name):
        """
            From Tumor WSIs extract negative patches from Normal area (reject tumor area)
            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param tumor_gt_mask:
            :param image_open: morphological open image of wsi_image
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to tumor regions
            :param patch_save_dir: directory to save patches into
            :param patch_prefix: prefix for patch name
            :param patch_index:
            :return:

        """

        mag_factor = pow(2, level_used)
#        tumor_gt_mask = cv2.cvtColor(tumor_gt_mask, cv2.COLOR_BGR2GRAY)
        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))
        bounding_box_num = 0
        for bounding_box in bounding_boxes:
            
            bounding_box_num += 1

            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            X = np.random.random_integers(b_x_start, high=b_x_end, size=NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)
            
#            print(np.shape(tumor_gt_mask))
#            print(np.shape(image_open))

#            txt_list_total = []
            for x, y in zip(X, Y):
                if int(image_open[y-1, x-1]) is not PIXEL_BLACK and int(tumor_gt_mask[y-1, x-1]) is not PIXEL_WHITE:#tissue but not tumor
                    # mask_gt does not contain tumor area
#                    print('coord (%d,%d) with %d th bounding_box'%(x,y,bounding_box_num))
#                    patch = wsi_image.read_region((x * mag_factor, y * mag_factor), 0,
#                                                  (PATCH_SIZE, PATCH_SIZE))
#                    patch.save(patch_save_dir + patch_prefix + str(patch_index), 'jpeg')
#                    patch.save(patch_save_dir + str(patch_index), 'jpeg')
                    txt_list = wsi_name+','+ str((x-1) * mag_factor + PATCH_SIZE/2) + ',' + str((y-1) * mag_factor + PATCH_SIZE/2)
#                    write_to_txt(txt_list, "Resample_normal_valid.txt")
                    write_to_txt(txt_list, "Resample_negative.txt")
#                    txt_list_total.append(txt_list)
                    patch_index += 1
#                    patch.close()

        return patch_index#, txt_list_total


class WSIOps(object):
    """
        # ================================
        # Class to annotate WSIs with ROIs
        # ================================

    """

    def_level = 0

    @staticmethod
    def read_wsi_mask(mask_path, level=def_level):
        try:
            wsi_mask = OpenSlide(mask_path)

            mask_image = np.array(wsi_mask.read_region((0, 0), level,
                                                       wsi_mask.level_dimensions[level]))

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None

        return wsi_mask, mask_image

    @staticmethod
    def read_wsi_normal(wsi_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            wsi_image = OpenSlide(wsi_path)
            level_used = wsi_image.level_count - 1
#            level_used = 0
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None, None

        return wsi_image, rgb_image, level_used

    @staticmethod
    def read_wsi_tumor(wsi_path, mask_path):
#    def read_wsi_tumor(wsi_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            wsi_image = OpenSlide(wsi_path)
#            print('loading wsi image done')
            wsi_mask = np.load(mask_path, mmap_mode='r')
            print('loading wsi mask done')

#            level_used = wsi_image.level_count - 1
            level_used = 6
            ###载入特别慢
#            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
#                                                       wsi_image.level_dimensions[level_used]))
            rgb_image = wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used])
            print('loading wsi image done')
                                                       
#            mask_level = wsi_mask.level_count - 1
#            tumor_gt_mask = wsi_mask.read_region((0, 0), mask_level,
#                                                 wsi_image.level_dimensions[mask_level])
#            resize_factor = float(1.0 / pow(2, level_used - mask_level))
            # print('resize_factor: %f' % resize_factor)
#            tumor_gt_mask = cv2.resize(wsi_mask, (0, 0), fx=resize_factor, fy=resize_factor)
            tumor_gt_mask = wsi_mask
            
#            wsi_mask.close()
        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None, None, None

        return wsi_image, rgb_image, wsi_mask, tumor_gt_mask, level_used
#        return wsi_image, wsi_mask, level_used
#        return wsi_image, rgb_image, level_used

    def read_wsi(wsi_path, mask_path, negative=True):
#    def read_wsi_tumor(wsi_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
#        print('wsi_path',wsi_path)
        try:
#            print(wsi_path)
            wsi_image = OpenSlide(wsi_path)
            level_used = 6
            print(wsi_path)
            print(mask_path)
#            print('loading wsi image done')
            if np.size(mask_path):
                tumor_gt_mask = np.load(mask_path, mmap_mode='r')

                ##tumor numpy is in level 0 
                ##need to be downsampled
#                tumor_gt_mask = tumor_gt_mask[:,range(0,tumor_gt_mask.shape[1],pow(2,level_used))]
#                tumor_gt_mask = tumor_gt_mask[range(tumor_gt_mask.shape[0],pow(2,level_used)),:]

#                tumor_gt_mask = tumor_gt_mask[level_used]
#                print('loading wsi mask done')
            else:
                tumor_gt_mask = []
            
            if negative:
                rgb_image = wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used])
#                print('loading wsi image done')
            else:
                rgb_image=[]
            
            
        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None, None, None

        return wsi_image, rgb_image, tumor_gt_mask, level_used
        
    def find_roi_bbox_tumor_gt_mask(self, mask_image):
        mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        bounding_boxes, _ = self.get_bbox(np.array(mask))
        return bounding_boxes

    def find_roi_bbox(self, rgb_image):
        # hsv -> 3 channel
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([200, 200, 200])
        # mask -> 1 channel
        mask = cv2.inRange(hsv, lower_red, upper_red)

        close_kernel = np.ones((20, 20), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
        bounding_boxes, rgb_contour = self.get_bbox(image_open, rgb_image=rgb_image)
        return bounding_boxes, rgb_contour, image_open

    @staticmethod
    def get_image_open(wsi_path):
        try:
            wsi_image = OpenSlide(wsi_path)
            level_used = 6
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))
            wsi_image.close()
        except OpenSlideUnsupportedFormatError:
            raise ValueError('Exception: OpenSlideUnsupportedFormatError for %s' % wsi_path)

        # hsv -> 3 channel
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([200, 200, 200])
        # mask -> 1 channel
        mask = cv2.inRange(hsv, lower_red, upper_red)

        close_kernel = np.ones((20, 20), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

        return image_open

    @staticmethod
    def get_bbox(cont_img, rgb_image=None):
        _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb_contour = None
        if rgb_image:
            rgb_contour = rgb_image.copy()
            line_color = (255, 0, 0)  # blue color code
            cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        return bounding_boxes, rgb_contour

    @staticmethod
    def draw_bbox(image, bounding_boxes):
        rgb_bbox = image.copy()
        for i, bounding_box in enumerate(bounding_boxes):
            x = int(bounding_box[0])
            y = int(bounding_box[1])
            cv2.rectangle(rgb_bbox, (x, y), (x + bounding_box[2], y + bounding_box[3]), color=(0, 0, 255),
                          thickness=2)
        return rgb_bbox

    @staticmethod
    def split_bbox(image, bounding_boxes, image_open):
        rgb_bbox_split = image.copy()
        for bounding_box in bounding_boxes:
            for x in range(bounding_box[0], bounding_box[0] + bounding_box[2]):
                for y in range(bounding_box[1], bounding_box[1] + bounding_box[3]):
                    if int(image_open[y, x]) == 1:
                        cv2.rectangle(rgb_bbox_split, (x, y), (x, y),
                                      color=(255, 0, 0), thickness=2)

        return rgb_bbox_split
