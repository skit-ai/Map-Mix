import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(input, target, size_average=True):
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss

class CrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target):
        return cross_entropy_loss(input, target, self.size_average)


from matplotlib.pyplot import ylabel
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from IPython import embed
from netcal.metrics import ECE

def EER(y, y_softmax_scores, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]) :
    y = label_binarize(y, classes=classes)
    n_classes = 14
    y_softmax_scores = np.stack(y_softmax_scores, axis=0)
    total_eer = 0
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y[:, i], y_softmax_scores[:, i])
            fnr = 1 - tpr
            eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            total_eer += eer
        except:
            pass
    
    average_eer = total_eer/n_classes
    return average_eer

def Cavg(y, y_pred):
    # https://www.nist.gov/system/files/documents/2017/09/29/lre17_eval_plan-2017-09-29_v1.pdf
    # section 3.1
    ntar = 14
    cavg_1 = 0.0
    beta_1 = 1.0

    cavg_2 = 0.0
    beta_2 = 9.0

    P_FA = np.zeros((ntar,ntar)).astype(float)
    P_Miss = np.zeros((ntar,)).astype(float)

    for i, label in enumerate(y):
        pred_label = y_pred[i]
        if(label != pred_label):
            P_FA[label][pred_label] += 1
            P_Miss[label] += 1

    tar_count = np.array([y.count(i) for i in range(ntar)])
    print(P_FA)
    print(tar_count)

    # get probabilities
    P_FA /= tar_count.reshape(1, -1).transpose()
    P_Miss /= tar_count

    print(P_FA)
    print(P_Miss)

    for i in range(ntar):
        cavg_1 += P_Miss[i]
        cavg_2 += P_Miss[i]
        
        for j in range(ntar):
            cavg_1 += (beta_1/(ntar-1))*P_FA[i][j]
            cavg_2 += (beta_2/(ntar-1))*P_FA[i][j]

    cavg_1 /= ntar
    cavg_2 /= ntar
    
    c_primary = (cavg_1 + cavg_2)/2
    return c_primary

def ECEMetric(y, y_softmax_scores):
    y_softmax_scores = np.stack(y_softmax_scores, axis=0)
    ece = ECE(10)
    return ece.measure(y_softmax_scores, y)