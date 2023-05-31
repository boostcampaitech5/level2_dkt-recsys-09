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

        grouped_df = df.groupby(col_name)
        
        mean_series = grouped_df.mean()['answerCode']
        std_series = grouped_df.std()['answerCode']
        sum_series = grouped_df.sum()['answerCode']
        
        
        series2mean = dict()
        for i, v in zip(mean_series.keys(), mean_series.values):
            series2mean[i] = v
            
        series2std = dict()
        for i, v in zip(std_series.keys(), std_series.values):
            series2std[i] = v
            
        series2sum = dict()
        for i, v in zip(sum_series.keys(), sum_series.values):
            series2sum[i] = v
            
        df[f'{new_feature_name}_ans_mean'] = df[col_name].map(series2mean)
        df[f'{new_feature_name}_ans_std'] = df[col_name].map(series2std)
        df[f'{new_feature_name}_ans_sum'] = df[col_name].map(series2sum)
        
        return df


    ## col_name를 기준으로 mean, std, sum을 추가하는 함수.
    def new_feature_answer(df, col_name:str, new_feature_name:str): 

        grouped_df = df.groupby(col_name)
        
        mean_series = grouped_df.mean()['answerCode']
        std_series = grouped_df.std()['answerCode']
        sum_series = grouped_df.sum()['answerCode']
        
        
        series2mean = dict()
        for i, v in zip(mean_series.keys(), mean_series.values):
            series2mean[i] = v
            
        series2std = dict()
        for i, v in zip(std_series.keys(), std_series.values):
            series2std[i] = v
            
        series2sum = dict()
        for i, v in zip(sum_series.keys(), sum_series.values):
            series2sum[i] = v
            
        df[f'{new_feature_name}_ans_mean'] = df[col_name].map(series2mean)
        df[f'{new_feature_name}_ans_std'] = df[col_name].map(series2std)
        df[f'{new_feature_name}_ans_sum'] = df[col_name].map(series2sum)
        
        return df
        
        
    # 난이도 설정을 위한 ELO 사용
    def get_ELO_function(df):
        def get_new_theta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
            return theta + learning_rate_theta(nb_previous_answers) * (
                is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
            )

        def get_new_beta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
            return beta - learning_rate_beta(nb_previous_answers) * (
                is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
            )

        def learning_rate_theta(nb_answers):
            return max(0.3 / (1 + 0.01 * nb_answers), 0.04)

        def learning_rate_beta(nb_answers):
            return 1 / (1 + 0.05 * nb_answers)

        def probability_of_good_answer(theta, beta, left_asymptote):
            return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def estimate_parameters(answers_df, granularity_feature_name="assessmentItemID"):
            item_parameters = {
                granularity_feature_value: {"beta": 0, "nb_answers": 0}
                for granularity_feature_value in np.unique(
                    answers_df[granularity_feature_name]
                )
            }
            student_parameters = {
                student_id: {"theta": 0, "nb_answers": 0}
                for student_id in np.unique(answers_df.userID)
            }

            print("Parameter estimation is starting...")

            for student_id, item_id, left_asymptote, answered_correctly in tqdm.tqdm(
                zip(
                    answers_df.userID.values,
                    answers_df[granularity_feature_name].values,
                    answers_df.left_asymptote.values,
                    answers_df.answerCode.values,
                )
            ):
                theta = student_parameters[student_id]["theta"]
                beta = item_parameters[item_id]["beta"]

                item_parameters[item_id]["beta"] = get_new_beta(
                    answered_correctly,
                    beta,
                    left_asymptote,
                    theta,
                    item_parameters[item_id]["nb_answers"],
                )
                student_parameters[student_id]["theta"] = get_new_theta(
                    answered_correctly,
                    beta,
                    left_asymptote,
                    theta,
                    student_parameters[student_id]["nb_answers"],
                )

                item_parameters[item_id]["nb_answers"] += 1
                student_parameters[student_id]["nb_answers"] += 1

            print(f"Theta & beta estimations on {granularity_feature_name} are completed.")
            return student_parameters, item_parameters

        def gou_func(theta, beta):
            return 1 / (1 + np.exp(-(theta - beta)))

        df["left_asymptote"] = 0

        print(f"Dataset of shape {df.shape}")
        print(f"Columns are {list(df.columns)}")

        student_parameters, item_parameters = estimate_parameters(df)

        prob = [
            gou_func(student_parameters[student]["theta"], item_parameters[item]["beta"])
            for student, item in zip(df.userID.values, df.assessmentItemID.values)
        ]

        df["elo_prob"] = prob

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
    df = get_ELO_function(df)
    df = get_mission_feature(df)
    df = get_user_mean(df)
    
    df = new_feature_answer(df, 'testId', 'test')
    df = new_feature_answer(df, 'KnowledgeTag', 'tag')
    df = new_feature_answer(df, 'prefix', 'prefix')
    df = new_feature_answer(df, 'assessmentItemID', 'assess')
    
    df['recent3_elap_time'] = df.groupby(['userID'])['elap_time'].rolling(3).mean().fillna(0).values
    
    
    # time_df = df[["userID", "prefix", "Timestamp"]].sort_values(by=["userID", "prefix", "Timestamp"])
    # time_df["first"] = time_df[["userID_reset", "prefix_reset"]].any(axis=1).apply(lambda x: 1 - int(x))
    # time_df["reset_time"] = time_df["Timestamp"].diff().fillna(pd.Timedelta(seconds=0))
    # time_df["reset_time"] = (
    #     time_df["reset_time"].apply(lambda x: x.total_seconds()) * time_df["first"]
    # )
    # df["reset_time"] = time_df["reset_time"]#.apply(lambda x: math.log(x + 1))
    
    # time_df["reset_time"] = time_df["Timestamp"].diff().fillna(pd.Timedelta(seconds=0))
    # time_df["reset_time"] = (
    #     time_df["reset_time"].apply(lambda x: x.total_seconds()) * time_df["first"]
    # )
    # df["reset_time"] = time_df["reset_time"]#.apply(lambda x: math.log(x + 1))
    
    return df
    
    

