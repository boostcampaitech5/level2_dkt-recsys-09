import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold



def fe(df):

    ## col_name를 기준으로 mean, std, sum을 추가하는 함수.
    def new_feature_answer(df, col_name:str, new_feature_name:str): 

        mean_series = df.groupby(col_name).agg({'answerCode':'mean'}).to_dict()['answerCode']
        std_series = df.groupby(col_name).agg({'answerCode':'std'}).to_dict()['answerCode']
        sum_series = df.groupby(col_name).agg({'answerCode':'sum'}).to_dict()['answerCode']
            
        df[f'{new_feature_name}_ans_mean'] = df[col_name].map(mean_series)
        df[f'{new_feature_name}_ans_std'] = df[col_name].map(std_series)
        df[f'{new_feature_name}_ans_sum'] = df[col_name].map(sum_series)
        
        return df
    
    
    def get_elap_time(df):
        solving_time = df[['userID', 'Timestamp']].groupby('userID').diff(periods=-1).fillna(pd.Timedelta(seconds=0))
        solving_time = solving_time['Timestamp'].apply(lambda x: x.total_seconds())
        df['elap_time'] = -solving_time
        df['elap_time'] = df['elap_time'].map(lambda x: int(x) if 0 < x <= 3600 else int(89))

        elap_mean_time = df[['assessmentItemID', 'elap_time']].groupby('assessmentItemID').mean().rename(columns={'elap_time': 'elap_mean_time'})
        elap_median_time = df[['assessmentItemID', 'elap_time']].groupby('assessmentItemID').median().rename(columns={'elap_time': 'elap_median_time'})
        df = pd.merge(df, elap_mean_time, on='assessmentItemID', how='left')
        df = pd.merge(df, elap_median_time, on='assessmentItemID', how='left')
        return df
        
        
    def get_mission_feature(df):
        #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        df.sort_values(by=['userID','Timestamp'], inplace=True)
        
        #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
        df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
        df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
        df['user_correct_answer'].iloc[0] = 0 # fill first Nan to 0
        df['user_acc'].iloc[0] = 0 # fill first Nan to 0

        # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
        # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
        correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum', 'std'])
        correct_t.columns = ["test_mean", 'test_sum', 'test_std']
        correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum', 'std'])
        correct_k.columns = ["tag_mean", 'tag_sum', 'tag_std']

        df = pd.merge(df, correct_t, on=['testId'], how="left")
        df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
        return df
    
    def get_user_mean(df):
        stu_groupby = df.groupby('userID').agg({
        'assessmentItemID': 'count',
        'answerCode': 'sum'
		  })
        stu_groupby['user_mean'] = stu_groupby['answerCode'] / stu_groupby['assessmentItemID']
        stu_groupby = stu_groupby.reset_index()
        df = df.merge(stu_groupby[['userID','user_mean']], on='userID', how='left')
        return df
    
    
    # create prefix, suffix
    df['prefix'] = df.assessmentItemID.map(lambda x: int(x[2:3]))
    df['suffix'] = df.assessmentItemID.map(lambda x: int(x[-3:]))
    
    # create elap_time, ELO, mission' featurem, user_mean
    df = get_elap_time(df)
    df = get_mission_feature(df)
    df = get_user_mean(df)
    
    df = new_feature_answer(df, 'testId', 'test')
    df = new_feature_answer(df, 'KnowledgeTag', 'tag')
    df = new_feature_answer(df, 'prefix', 'prefix')
    df = new_feature_answer(df, 'assessmentItemID', 'assess')
    
    df['recent3_elap_time'] = df.groupby(['userID'])['elap_time'].rolling(3).mean().fillna(0).values
    
    return df