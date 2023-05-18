import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def accuracy(output, target):
    with torch.no_grad():
        pred = output.round()
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def auc(output, target):
    y_pred = output.detach().cpu().numpy()
    y = target.int().detach().cpu().numpy()
    
    return roc_auc_score(y, y_pred)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
