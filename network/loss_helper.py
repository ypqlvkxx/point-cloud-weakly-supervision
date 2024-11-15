import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def compute_supervised_loss(output, label, dataset, class_weights):
    CE_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    logits = output['logits']
    logits = logits.transpose(1, 2).reshape(-1, dataset.num_classes)
    labels = label.reshape(-1)
    # Boolean mask of points that should be ignored
    ignored_bool = (labels == 0)
    #
    for ign_label in dataset.ignored_labels:
        ignored_bool = ignored_bool | (labels == ign_label)
    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits = logits[valid_idx, :]
    valid_labels_init = labels[valid_idx]
    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, dataset.num_classes).long().to(logits.device)
    inserted_value = torch.zeros((1,)).long().to(logits.device)
    for ign_label in dataset.ignored_labels:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
    loss = CE_loss(valid_logits, valid_labels).mean()
    output['valid_logits'], output['valid_labels'] = valid_logits, valid_labels
    output['loss'] = loss
    return loss

def compute_Entropy_regularization_loss(output, mask=None, reduction='mean'):
    logits = output['logits']
    output_reshaped = logits.permute(0,2,1).softmax(-1)[mask, :]
    #softmax
    probs = output_reshaped 
    #
    entropy_loss = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  #
    #
    if reduction == 'none':  
        return entropy_loss  
    elif reduction == 'mean':  
        return torch.mean(entropy_loss)  
    elif reduction == 'sum':  
        return torch.sum(entropy_loss)  
    
    return entropy_loss

def compute_consistency_loss(output, ensemble_prediction, mask=None, reduction='mean'):
    consistency_loss = nn.MSELoss() 
    logits = output['logits']
    output_reshaped = logits.permute(0,2,1).softmax(-1)[mask, :]
    
    es_probs = []
    for id_ in zip(output['idx'].cpu().numpy()):  
        es_probs.append(torch.from_numpy(ensemble_prediction[id_[0][0]]))
    es_probs = torch.stack(es_probs)
    es_probs = es_probs.cuda(non_blocking=True)
    #softmax
    probs = output_reshaped 
    es_probs = es_probs.permute(0,2,1)[mask,:]
    loss = consistency_loss(probs, es_probs)
    
    return loss