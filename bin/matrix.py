# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true > 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc: \t': acc,
                'Mean Acc : \t': acc_cls,
                'FreqW Acc : \t': fwavacc,
                'Mean IoU : \t': mean_iu,}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

def pixel_wise_accuracy_numpy(pred, mask):
    '''pred and mask are two numpy arrays'''
    acc = float(np.count_nonzero(mask==pred))/mask.size
    return acc

def dice_score_numpy(pred, mask):
    '''pred and mask are two numpy arrays'''
    if np.sum(mask) == 0.0: # a negative patch
        mask = (mask == 0).astype(np.uint8)
        pred = (pred == 0).astype(np.uint8)
    inter = np.sum(mask[pred == 1]) + 0.000001
    union = np.sum(pred) + np.sum(mask) + 0.00001
    dice = 2*inter / union
    #if dice = 2.0: dice = 1.0
    return dice

def tp_fp_tn_fn(pred,mask):
    '''pred and mask are two numpy arrays'''
    tp = np.sum(mask[pred == 1])
    fn = np.sum(mask[pred == 0])
    fp = np.sum(pred[mask == 0])
    tn = mask.size - tp - fn - fp



    #print tp,tn,fp,fn
    #print TPR, TNR, PPV, FPV, ACC
    return tp, fp, tn, fn

def get_score(tp,fp,tn,fn):
    TPR = float(tp) / (tp + fn)
    #TNR = float(tn) / (tn + fp)
    PPV = float(tp) / (tp + fp)
    #FPV = float(tn) / (tn + fn)
    ACC = float(tp + tn) / (tp + fp + fn + tn)
    F1_Score = 2 * TPR * PPV / (TPR + PPV)

    return TPR, ACC, F1_Score

class confusion_matrix(object):
    def __init__(self, nclass):
        self.conf = np.ndarray((nclass,nclass),dtype=np.int32)
        self.nclass = nclass
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self,predicted, target):
        '''predicted and target could be two arrays'''
        predicted = np.array(predicted)
        target = np.array(target)
        x = predicted + self.nclass * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.nclass**2)
        conf = bincount_2d.reshape((self.nclass,self.nclass))
        self.conf += conf

    def value(self):
        '''row = gt, col = predict
                 be predicted
                     0      1
        target 0     2      3
               1     4      5
        there are 2 "0" predicted as "0", 3 "0" predicted as "1"(fp)
        there are 4 "1" predicted as "0"(fn), 5 "1" predicted as "1"
        mAP = (2+5)/(2+3+4+5)

        '''
        return self.conf
