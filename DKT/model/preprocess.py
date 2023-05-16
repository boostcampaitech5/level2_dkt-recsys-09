import pandas as pd
import os
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime
import joblib
from pickle import dump
from pickle import load

def load_data(args):
    csv_file_path = os.path.join(args.data_dir, 'train_data.csv') 
    df = pd.read_csv(csv_file_path) 
    return df


def feature_engineering(df):
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    
    #유저들의 문제 풀이수, 정답 수, 정답률
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']

    #testId와 KnowledgeTag의 전체 정답률
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
    correct_t.columns = ["test_mean", 'test_sum']
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    correct_k.columns = ["tag_mean", 'tag_sum']

    df = pd.merge(df, correct_t, on=['testId'], how="left")
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    
    return df


def categorical_label_encoding(args, df, is_train=True):
    cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

    if not os.path.exists(args.asset_dir):
        os.makedirs(args.asset_dir)    

    for col in cate_cols:
        le = LabelEncoder()
        if is_train:
            # For UNKNOWN class
            a = df[col].unique().tolist() + ["unknown"]
            le.fit(a)
            le_path = os.path.join(args.asset_dir, col + "_classes.npy")            
            np.save(le_path, le.classes_)
        else:
            label_path = os.path.join(args.asset_dir, col + "_classes.npy")
            le.classes_ = np.load(label_path)
            df[col] = df[col].apply(lambda x: x if str(x) in le.classes_ else "unknown")

        # 모든 컬럼이 범주형이라고 가정
        df[col] = df[col].astype(str)
        test = le.transform(df[col])
        df[col] = test

    return df

def convert_time(s):
     timestamp = time.mktime(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())
     return int(timestamp)


def add_diff_feature(df):
    df = df.sort_values(['userID', 'Timestamp'])

    # diff
    df['diff'] = df.sort_values(['userID','Timestamp']).groupby('userID')['Timestamp'].diff()

    diff_df = df['diff']
    diff_df.dropna(inplace=True)

    # nan은 -1
    # 600(10분) 이상이면 다 600
    df['diff'].fillna(-1, inplace=True)
    idx = df[df['diff'] >= 600].index
    df.loc[idx, 'diff'] = 600

    tmp = df[df['diff'] >= 0]
    correct_k = tmp.groupby(['KnowledgeTag'])['diff'].agg(['mean'])
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    return df


def scaling(args, df, is_train=True):
    columns = ['user_total_answer', 'user_correct_answer', 'Timestamp', 'test_sum', 'tag_sum']
    
    for col in columns:
        if is_train:
            scaler = MinMaxScaler()
            scaler.fit(df[col].values.reshape(-1, 1))
            sc_path = os.path.join(args.asset_dir, col + "_scaler.pkl")
            dump(scaler, open(sc_path, 'wb'))
        else:
            sc_path = os.path.join(args.asset_dir, col + "_scaler.pkl")
            scaler = load(open(sc_path, 'rb'))
        
        df[col] = scaler.transform(df[col].values.reshape(-1, 1))
    
    return df


# train과 test 데이터셋은 사용자 별로 묶어서 분리를 해주어야함
def custom_train_test_split(args, df, split=True):
    random.seed(args.seed)
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    random.shuffle(users)
    
    max_train_data_len = args.split_ratio*len(df)
    sum_of_train_data = 0
    user_ids =[]

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)


    train = df[df['userID'].isin(user_ids)]
    test = df[df['userID'].isin(user_ids) == False]

    #test데이터셋은 각 유저의 마지막 interaction만 추출
    #test = test[test['userID'] != test['userID'].shift(-1)]
    
    # 결측치 채우기
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    
    return train, test 