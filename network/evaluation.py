import numpy as np

def compute_iou(cm, class_weight, ignore_zero=False):
    cm = cm.cpu().detach().numpy()
    if cm.sum() == 0: return 0, 0

    tp = np.diag(cm)
    with np.errstate(divide='ignore'):
        ciou = tp / (cm.sum(1) + cm.sum(0) - tp)
    if ignore_zero:
        ciou = ciou[1:]
    for i in range(len(ciou)):
        if np.isnan(ciou[i]):
            ciou[i] = 0.0
    count = 0
    miou = 0
    for w, iou in zip(class_weight[0,:].cpu().numpy(), ciou):  
        if w != 0:  # 
            miou += iou  # 
            count += 1  #

    miou = (miou / count) * 100
    return ciou, miou, cm

def compute_oa(cm, ignore_zero=False):
    cm = cm.cpu().detach().numpy()
    if cm.sum() == 0: return 0, 0

    

    tp, fp, fn, tn = 0, 0, 0, 0
    tp = np.diag(cm)
    fp = np.sum(cm, axis=1) - tp
    fn = np.sum(cm, axis=0) - tp
    tn = np.sum(cm) - (np.sum(cm, axis=0) + np.sum(cm, axis=1) - tp)
    
    oa_m = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    oa = 0
    count = 0
    with np.errstate(divide='ignore'):
        ciou = tp / (cm.sum(1) + cm.sum(0) - tp)
    if ignore_zero:
        ciou = ciou[1:]
    for i in range(len(ciou)):
        oa += oa_m[i]
        count += 1
        if np.isnan(ciou[i]):
            oa -= oa_m[i]
            count -= 1
    oa = oa / count
    #miou = np.nanmean(ciou) * 100
    acc_global = np.diag(cm).sum() / np.sum(cm)
    return oa, acc_global