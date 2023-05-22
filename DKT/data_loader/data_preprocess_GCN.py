import pandas as pd
import time
import datetime
import pickle
import torch
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np


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
    
    with open('constraint_matrix.pickle', 'wb') as f:
        pickle.dump(constraint_mat, f)
        

def hybrid_preprocess(data_dir, args):
    df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    df = __preprocessing(df)

    # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
    args.n_questions = df['assessmentItemID'].nunique()
    args.n_test = df['testId'].nunique()
    args.n_tag = df['KnowledgeTag'].nunique()

    df = df.sort_values(by=['userID','Timestamp'], axis=0)
    columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag']
    group = df[columns].groupby('userID').apply(
            lambda r: (
                r['testId'].values, 
                r['assessmentItemID'].values,
                r['KnowledgeTag'].values,
                r['answerCode'].values
            )
        )

def __save_labels(encoder, name, args):
        le_path = os.path.join(args.data_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)


def __preprocessing(df, args):
    cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag']
    for col in cate_cols:

        #For UNKNOWN class
        a = df[col].unique().tolist() + [np.nan]
        
        le = LabelEncoder()
        le.fit(a)
        df[col] = le.transform(df[col])
        __save_labels(le, col, args)

    def convert_time(s):
        timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
        return int(timestamp)

    df['Timestamp'] = df['Timestamp'].apply(convert_time)
    
    return df