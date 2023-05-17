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
import math

def load_data(args):
    csv_file_path = os.path.join(args.data_dir, args.df_name)
    df = pd.read_csv(csv_file_path) 
    return df


def feature_engineering(df):
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


    df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
    df['dow'] = pd.to_datetime(df['Timestamp']).dt.dayofweek # 요일을 숫자로

    df['testcode']=df['testId'].apply(lambda x : int(x[1:4])//10)
    df['problem_number'] = df['assessmentItemID'].apply(lambda x: int(x[7:])) 

    # feature 별 정답여부
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
    correct_t.columns = ["test_mean", 'test_sum']
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    correct_k.columns = ["tag_mean", 'tag_sum']
    correct_a = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum'])
    correct_a.columns = ["ass_mean", 'ass_sum']
    correct_p = df.groupby(['problem_number'])['answerCode'].agg(['mean', 'sum'])
    correct_p.columns = ["prb_mean", 'prb_sum']
    correct_h = df.groupby(['hour'])['answerCode'].agg(['mean', 'sum'])
    correct_h.columns = ["hour_mean", 'hour_sum']
    correct_d = df.groupby(['dow'])['answerCode'].agg(['mean', 'sum'])
    correct_d.columns = ["dow_mean", 'dow_sum'] 

    df = pd.merge(df, correct_t, on=['testId'], how="left")
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, correct_a, on=['assessmentItemID'], how="left")
    df = pd.merge(df, correct_p, on=['problem_number'], how="left")
    df = pd.merge(df, correct_h, on=['hour'], how="left")
    df = pd.merge(df, correct_d, on=['dow'], how="left")


    f = lambda x : len(set(x))
    t_df = df.groupby(['testId']).agg({
    'problem_number':'max',
    'KnowledgeTag':f
    })
    t_df.reset_index(inplace=True)

    t_df.columns = ['testId','problem_count',"tag_count"]

    df = pd.merge(df,t_df,on='testId',how='left')

    gdf = df[['userID','testId','problem_number','testcode','Timestamp']].sort_values(by=['userID','testcode','Timestamp'])
    gdf['buserID'] = gdf['userID'] != gdf['userID'].shift(1)
    gdf['btestcode'] = gdf['testcode'] != gdf['testcode'].shift(1)
    gdf['first'] = gdf[['buserID','btestcode']].any(axis=1).apply(lambda x : 1- int(x))
    gdf['RepeatedTime'] = pd.to_datetime(gdf['Timestamp']).diff().fillna(pd.Timedelta(seconds=0)) 
    gdf['RepeatedTime'] = gdf['RepeatedTime'].apply(lambda x: x.total_seconds()) * gdf['first']
    df['RepeatedTime'] = gdf['RepeatedTime'].apply(lambda x : math.log(x+1))

    df['prior_KnowledgeTag_frequency'] = df.groupby(['userID','KnowledgeTag']).cumcount()

    df['problem_position'] = df['problem_number'] / df["problem_count"]
    df['solve_order'] = df.groupby(['userID','testId']).cumcount()
    df['solve_order'] = df['solve_order'] - df['problem_count']*(df['solve_order'] > df['problem_count']).apply(int) + 1
    df['retest'] = (df['solve_order'] > df['problem_count']).apply(int)
    T = df['solve_order'] != df['problem_number']
    TT = T.shift(1)
    TT[0] = False
    df['solved_disorder'] = (TT.apply(lambda x : not x) & T).apply(int)

    df['testId'] = df['testId'].apply(lambda x : int(x[1:4]+x[-3]))

    # 정답과 오답 기준으로 나눠서 생각
    o_df = df[df['answerCode']==1]
    x_df = df[df['answerCode']==0]

    diff_k = df.groupby(['KnowledgeTag'])['diff'].agg('mean').reset_index()
    diff_k.columns = ['KnowledgeTag',"tag_diff"]
    diff_k_o = o_df.groupby(['KnowledgeTag'])['diff'].agg('mean').reset_index()
    diff_k_o.columns = ['KnowledgeTag', "tag_diff_o"]
    diff_k_x = x_df.groupby(['KnowledgeTag'])['diff'].agg('mean').reset_index()
    diff_k_x.columns = ['KnowledgeTag', "tag_diff_x"]

    df = pd.merge(df, diff_k, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, diff_k_o, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, diff_k_x, on=['KnowledgeTag'], how="left")

    ass_k = df.groupby(['assessmentItemID'])['diff'].agg('mean').reset_index()
    ass_k.columns = ['assessmentItemID',"ass_diff"]
    ass_k_o = o_df.groupby(['assessmentItemID'])['diff'].agg('mean').reset_index()
    ass_k_o.columns = ['assessmentItemID',"ass_diff_o"]
    ass_k_x = x_df.groupby(['assessmentItemID'])['diff'].agg('mean').reset_index()
    ass_k_x.columns = ['assessmentItemID',"ass_diff_x"]

    df = pd.merge(df, ass_k, on=['assessmentItemID'], how="left")
    df = pd.merge(df, ass_k_o, on=['assessmentItemID'], how="left")
    df = pd.merge(df, ass_k_x, on=['assessmentItemID'], how="left")

    prb_k = df.groupby(['problem_number'])['diff'].agg('mean').reset_index()
    prb_k.columns = ['problem_number',"prb_diff"]
    prb_k_o = o_df.groupby(['problem_number'])['diff'].agg('mean').reset_index()
    prb_k_o.columns = ['problem_number',"prb_diff_o"]
    prb_k_x = x_df.groupby(['problem_number'])['diff'].agg('mean').reset_index()
    prb_k_x.columns = ['problem_number',"prb_diff_x"]

    df = pd.merge(df, prb_k, on=['problem_number'], how="left")
    df = pd.merge(df, prb_k_o, on=['problem_number'], how="left")
    df = pd.merge(df, prb_k_x, on=['problem_number'], how="left")

    
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
    
    # 결측치 채우기
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    
    return train, test 