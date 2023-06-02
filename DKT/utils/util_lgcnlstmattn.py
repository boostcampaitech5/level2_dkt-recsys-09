import os
import random

import numpy as np
import torch
import scipy.sparse as sp



def setSeeds(seed=42):

    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def get_adj_matrix(train_dict, rel_dict, num_item, alpha, beta, max_len):
    row_seq = [train_dict[u][-max_len:][n] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)] + [train_dict[u][-max_len:][n+1] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)]
    col_seq = [train_dict[u][-max_len:][n+1] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)] + [train_dict[u][-max_len:][n] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)]

    row_sem = [i for i in rel_dict for j in rel_dict[i]] + [j for i in rel_dict for j in rel_dict[i]]
    col_sem = [j for i in rel_dict for j in rel_dict[i]] + [i for i in rel_dict for j in rel_dict[i]]

    rel_matrix = sp.coo_matrix(([alpha]*len(row_seq)+[beta]*len(row_sem), (row_seq+row_sem, col_seq+col_sem)), (num_item, num_item)).astype(np.float32) + sp.eye(num_item)
    
    row_sum = np.array(rel_matrix.sum(1)) + 1e-24
    degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
    rel_matrix_normalized = degree_mat_inv_sqrt.dot(rel_matrix.dot(degree_mat_inv_sqrt)).tocoo()
    

    indices = np.vstack((rel_matrix_normalized.row, rel_matrix_normalized.col))
    values = rel_matrix_normalized.data.astype(np.float32)
    shape = rel_matrix_normalized.shape
    
    return indices, values, shape

def get_adj_matrix_wo_rel(train_dict, num_item, alpha=1, max_len=20):
    row_seq = [train_dict[u][-max_len:][n] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)] + [train_dict[u][-max_len:][n+1] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)]
    col_seq = [train_dict[u][-max_len:][n+1] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)] + [train_dict[u][-max_len:][n] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)]

    rel_matrix = sp.coo_matrix(([alpha]*len(row_seq), (row_seq, col_seq)), (num_item, num_item)).astype(np.float32) + sp.eye(num_item)
    
    row_sum = np.array(rel_matrix.sum(1)) + 1e-24
    
    degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
    
    rel_matrix_normalized = degree_mat_inv_sqrt.dot(rel_matrix.dot(degree_mat_inv_sqrt)).tocoo()
    
    indices = np.vstack((rel_matrix_normalized.row, rel_matrix_normalized.col))
    
    values = rel_matrix_normalized.data.astype(np.float32)
    
    shape = rel_matrix_normalized.shape
    
    return indices, values, shape


def get_adj_matrix_wo_normarlize(train_dict, num_item, alpha=1, max_len=20):
    
    row_seq = [train_dict[u][-max_len:][n] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)] + [train_dict[u][-max_len:][n+1] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)]
    col_seq = [train_dict[u][-max_len:][n+1] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)] + [train_dict[u][-max_len:][n] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)]

    rel_matrix = sp.coo_matrix(([alpha]*len(row_seq), (row_seq, col_seq)), (num_item, num_item)).astype(np.float32) + sp.eye(num_item)
    
    rel_matrix = rel_matrix.tocoo()
    
    indices = np.vstack((rel_matrix.row, rel_matrix.col))
    
    values = rel_matrix.data.astype(np.float32)
    
    shape = rel_matrix.shape
    
    return indices, values, shape