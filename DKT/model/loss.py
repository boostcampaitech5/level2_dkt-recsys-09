import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def get_betas(model, users, items):
    user_degree = model.constraint_mat['user_degree'].to('cuda')
    item_degree = model.constraint_mat['item_degree'].to('cuda')
    
    weights = 1 + model.lambda_ * (1/user_degree[users]) * torch.sqrt((user_degree[users]+1)/(item_degree[items]+1))
    
    return weights
    
    
def cal_loss_L(beta_weight, output, target):
    
    loss = F.binary_cross_entropy(output, target.float(), weight=beta_weight, reduction='none')
    
    return loss.sum()


def norm_loss(model):
    loss = 0.0
    for parameter in model.parameters():
        loss += torch.sum(parameter ** 2)
    return loss / 2


def UltraGCN_loss(model, output, data, target):
    
    users = data[:, 0]
    items = data[:, 1]
    
    beta_weight = get_betas(model, users, items)
    
    loss = cal_loss_L(beta_weight, output, target) 
    loss += model.gamma * norm_loss(model)

    return loss