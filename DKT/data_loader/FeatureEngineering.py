import numpy as np
import pandas as pd
import math
import os
from tqdm import tqdm

def FE(df):
    # 문제별 풀이시간
    from tqdm import tqdm

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['diff_Timestamp'] = df['Timestamp'] - df.shift(1)['Timestamp']

    testId_df = df[~df.duplicated(['assessmentItemID'])].groupby('testId')
    testId2len = {}
    for testId, g_df in testId_df:
        testId2len[testId] = len(g_df)

    userID_df = df.groupby('userID')
    start_index_list = []
    second_index_list = []

    for userID, g_df in tqdm(userID_df):
        testId_df = g_df.groupby('testId')
        for testId, gg_df in testId_df:
            index_list = gg_df.index.tolist()
            start_index = 0
            if len(gg_df) <= testId2len[testId]:
                start_index_list += [index_list[start_index]]
                second_index_list += [index_list[start_index + 1]]
            else:
                div = len(gg_df) // testId2len[testId]
                for _ in range(div):
                    start_index_list += [index_list[start_index]]
                    second_index_list += [index_list[start_index + 1]]
                    start_index += testId2len[testId]

    df.loc[start_index_list, 'diff_Timestamp'] = df.loc[second_index_list, 'diff_Timestamp'].values
    df['elapsed'] = df['diff_Timestamp'].apply(lambda x: x.total_seconds() if not pd.isna(x) else np.nan)


    df['hour'] = df['Timestamp'].dt.hour
    df['dow'] = df['Timestamp'].dt.dayofweek # 요일을 숫자로

    diff = df.loc[:, ['userID','Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
    diff = diff.fillna(pd.Timedelta(seconds=0))
    diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

    # 문제별 풀이시간
    df['elapsed'] = diff
    df['elapsed'] = df['elapsed'].apply(lambda x : x if x <650 and x >=0 else 0)

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


    # 정답과 오답 기준으로 나눠서 생각
    o_df = df[df['answerCode']==1]
    x_df = df[df['answerCode']==0]

    elp_k = df.groupby(['KnowledgeTag'])['elapsed'].agg('mean').reset_index()
    elp_k.columns = ['KnowledgeTag',"tag_elp"]
    elp_k_o = o_df.groupby(['KnowledgeTag'])['elapsed'].agg('mean').reset_index()
    elp_k_o.columns = ['KnowledgeTag', "tag_elp_o"]
    elp_k_x = x_df.groupby(['KnowledgeTag'])['elapsed'].agg('mean').reset_index()
    elp_k_x.columns = ['KnowledgeTag', "tag_elp_x"]

    df = pd.merge(df, elp_k, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, elp_k_o, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, elp_k_x, on=['KnowledgeTag'], how="left")

    ass_k = df.groupby(['assessmentItemID'])['elapsed'].agg('mean').reset_index()
    ass_k.columns = ['assessmentItemID',"ass_elp"]
    ass_k_o = o_df.groupby(['assessmentItemID'])['elapsed'].agg('mean').reset_index()
    ass_k_o.columns = ['assessmentItemID',"ass_elp_o"]
    ass_k_x = x_df.groupby(['assessmentItemID'])['elapsed'].agg('mean').reset_index()
    ass_k_x.columns = ['assessmentItemID',"ass_elp_x"]

    df = pd.merge(df, ass_k, on=['assessmentItemID'], how="left")
    df = pd.merge(df, ass_k_o, on=['assessmentItemID'], how="left")
    df = pd.merge(df, ass_k_x, on=['assessmentItemID'], how="left")

    prb_k = df.groupby(['problem_number'])['elapsed'].agg('mean').reset_index()
    prb_k.columns = ['problem_number',"prb_elp"]
    prb_k_o = o_df.groupby(['problem_number'])['elapsed'].agg('mean').reset_index()
    prb_k_o.columns = ['problem_number',"prb_elp_o"]
    prb_k_x = x_df.groupby(['problem_number'])['elapsed'].agg('mean').reset_index()
    prb_k_x.columns = ['problem_number',"prb_elp_x"]

    df = pd.merge(df, prb_k, on=['problem_number'], how="left")
    df = pd.merge(df, prb_k_o, on=['problem_number'], how="left")
    df = pd.merge(df, prb_k_x, on=['problem_number'], how="left")

    # 누적합 - 주어진 데이터 이전/이후 데이터들을 포함하는 메모리를 feature로 포함시킴: Sequence Model을 사용하지 않고 일반적인 지도 학습 모델에서 사용하기 위함
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
    df['testcode_o'] = df.groupby(['userID','testcode'])['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['testcodeCount'] = df.groupby(['userID','testcode']).cumcount()
    df['testcodeAcc'] = df['testcode_o']/df['testcodeCount']
    df['tectcodeElp'] = df.groupby(['userID','testcode'])['elapsed'].transform(lambda x: x.cumsum().shift(1))
    df['testcodeMElp'] = df['tectcodeElp']/df['testcodeCount']



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
    gdf['RepeatedTime'] = gdf['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)) 
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
    df['hour'] = df['Timestamp'].dt.hour
    df['dow'] = df['Timestamp'].dt.dayofweek

    df['testCode_cat'] = df['testcode'].apply(lambda x : (x-1) // 3)

    #user의 test별 평균 정답률 'testCorrectMean'
    group = df.groupby(['userID', 'testId'])['answerCode'].mean() #group = series
    df = df.sort_values(by=['userID', 'testId']).reset_index(drop=True)

    idx_lst = list(group.index)
    i = 0 #idx_lst의 인덱싱을 위한 변수
    j = 0 #df_mean의 row 인덱싱을 위한 변수

    while (i < len(idx_lst)):
        user, test = idx_lst[i]
        while (df.iloc[j]['userID'] == user) & (df.iloc[j]['testId'] == test):
            df.loc[j, 'testCorrectMean'] = group[user, test]
            j += 1
            if j == len(df):
                break
        i += 1

    df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

    #user의 test 별 정답률 추세
    df['shift'] = df.groupby(['userID', 'testId'])['answerCode'].shift().fillna(0)
    df['test_past'] = df.groupby(['userID', 'testId'])['shift'].cumsum()
    df['test_count'] = df.groupby(['userID', 'testId']).cumcount()
    df['testCorrectRate'] = (df['test_past'] / df['test_count']).fillna(0)
    df.drop(['shift', 'test_past', 'test_count'], axis=1, inplace=True)

    return df

"""
# train test 각각 받고 
train_feature_engineered not 
    FE(train)
    저장

test_feature_engineered not 
    FE(test)
    저장
"""