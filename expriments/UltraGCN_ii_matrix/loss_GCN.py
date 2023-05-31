import torch.nn.functional as F
import torch
import os
import pickle

def nll_loss(output, target):
    return F.nll_loss(output, target)


def get_betas(model, users, items):
    user_degree = model.constraint_mat['user_degree'].to('cuda')
    item_degree = model.constraint_mat['item_degree'].to('cuda')
    
    weights = 1 + model.lambda_ * (1/user_degree[users]) * torch.sqrt((user_degree[users]+1)/(item_degree[items]+1))
    
    return weights

def get_omegas(model):
    ii_mat_idx = model.ii_constraint_idx_mat
    ii_mat_sim = model.ii_constraint_sim_mat
    ii_mat_diagonal = model.ii_constraint_diagonal_mat.to('cuda')
    
    g_i = torch.sum(ii_mat_sim, 1).to('cuda')
    ii_mat_idx.apply_(lambda x: g_i[int(x)].squeeze().item())
    
    ii_mat_sim = ii_mat_sim.to('cuda')
    ii_mat_idx = ii_mat_idx.to('cuda')
    
    weights = (ii_mat_sim / (g_i.unsqueeze(1).expand(-1, ii_mat_sim.shape[1]) - ii_mat_diagonal.unsqueeze(1).expand(-1, ii_mat_sim.shape[1]))) * torch.sqrt(g_i.unsqueeze(1).expand(-1, ii_mat_sim.shape[1]) / ii_mat_idx)
    
    return weights

    
def cal_loss_L(beta_weight, output, target):
    
    loss = F.binary_cross_entropy(output, target.float(), weight=beta_weight, reduction='none')
    
    return loss.sum()


def cal_loss_I(model, omega_weight, users, items):
    ii_mat_idx = model.ii_constraint_idx_mat.to('cuda')
    
    user_embeds = model.user_embeds
    item_embeds = model.item_embeds
    
    item_idx_mat = ii_mat_idx[items].squeeze(1)
    
    e_j = item_embeds(item_idx_mat.int())
    e_u = user_embeds(users)
    
    mm = torch.log((e_j * e_u).sum(-1).sigmoid())
    weight = omega_weight[items].squeeze(1)
    
    loss = (mm * weight).sum(-1)
    
    return -1 * loss.sum()


def norm_loss(model):
    loss = 0.0
    for parameter in model.parameters():
        loss += torch.sum(parameter ** 2)
    return loss / 2


def UltraGCN_loss(model, output, data, target):
    
    users = data[:, 0]
    items = data[:, 1]
    
    beta_weight = get_betas(model, users, items)
    
    if not os.path.exists("./matrix/omega.pickle"):
        with open('./matrix/omega.pickle', 'wb') as f:
            pickle.dump(get_omegas(model), f)
        
    with open('./matrix/omega.pickle', 'rb') as f:
        omega_weight = pickle.load(f)
    
    pos_idx = torch.nonzero(target)

    loss = cal_loss_L(beta_weight, output, target) 
    loss += cal_loss_I(model, omega_weight, users[pos_idx], items[pos_idx]) * model.delta
    loss += model.gamma * norm_loss(model)

    return loss