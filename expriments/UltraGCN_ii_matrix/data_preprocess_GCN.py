import pandas as pd
import pickle
import torch

def ultragcn_preprocess(train, test):
    
    # 한 유저가 같은 문제를 여러 번 푼 경우 마지막 성적만을 반영
    data = pd.concat([train, test]).drop_duplicates(subset = ["userID", "assessmentItemID"],
                                                    keep = "last")
    # userID, assessmentItemID, Timestamp indexing 진행
    data = _indexing(data)
    
    # answerCode가 -1인 항목 test data로 분리
    test_data = data[data.answerCode == -1]
    test_data.to_csv("~/input/data/test_data_modify.csv", index=False)
    
    data = data[data.answerCode >= 0]
    data.to_csv("~/input/data/data.csv", index=False)

    # 모델 학습 시 필요한 constraint matrix를 저장
    save_ii_constraint_matrix(data)
    save_constraint_matrix(data)
    

def _indexing(data):
    
    # userID와 itemID indexing
    userid, itemid = sorted(list(set(data.userID))), sorted(list(set(data.assessmentItemID)))

    userid_2_index = {v:i for i,v in enumerate(userid)}
    itemid_2_index = {v:i for i,v in enumerate(itemid)}
    
    data.userID = data.userID.map(userid_2_index)
    data.assessmentItemID = data.assessmentItemID.map(itemid_2_index)

    return data[['userID', 'assessmentItemID', 'answerCode']]


def save_constraint_matrix(data):
    
    user_groupby = data.groupby('userID').agg({'assessmentItemID':'count'}).sort_values('userID').assessmentItemID.to_list()
    item_groupby = data.groupby('assessmentItemID').agg({'userID':'count'}).sort_values('assessmentItemID').userID.to_list()

    constraint_mat = {"user_degree": torch.Tensor(user_groupby),
                      "item_degree": torch.Tensor(item_groupby)}
    
    with open('./matrix/constraint_matrix.pickle', 'wb') as f:
        pickle.dump(constraint_mat, f)

        
def save_ii_constraint_matrix(data):
    
    adj_df = data.pivot(index='userID', columns='assessmentItemID', values='answerCode').fillna(0)
    adj_matrix = torch.from_numpy(adj_df.values).float().to('cuda')
    
    num_neighbors = 10
    A = adj_matrix.T.matmul(adj_matrix)	# I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))

    for i in range(n_items):
        row = A[i, :]
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        
    with open('./matrix/ii_constraint_sim_matrix.pickle', 'wb') as f:
        pickle.dump(res_sim_mat, f)
        
    with open('./matrix/ii_constraint_idx_matrix.pickle', 'wb') as f:
        pickle.dump(res_mat, f)
        
    with open('./matrix/ii_constraint_diagonal_matrix.pickle', 'wb') as f:
        pickle.dump(torch.diagonal(A), f)