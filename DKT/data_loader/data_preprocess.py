import pandas as pd
import time
import datetime

def ultragcn_preprocess(train, test):
    
    # 한 유저가 같은 문제를 여러 번 푼 경우 마지막 성적만을 반영
    data = pd.concat([train, test]).drop_duplicates(subset = ["userID", "assessmentItemID"],
                                                    keep = "last")
    
    # answerCode가 -1인 항목 제거 (평가 항목 제거)
    data = data[data.answerCode >= 0].reset_index(drop=True)
    
    # userID, assessmentItemID, Timestamp indexing 진행
    data = _indexing(data)
    
    # 유저별 마지막 항목을 validation set으로 사용
    eval_data = data.copy()
    eval_data.drop_duplicates(subset = ["userID"],
                                  keep = "last", inplace = True)
    
    # validataion set을 제외한 data를 train set으로 사용
    train_data = data.drop(eval_data.index, axis=0)
    
    return data, eval_data.index, train_data.index
    

def _indexing(data):
    userid, itemid = sorted(list(set(data.userID))), sorted(list(set(data.assessmentItemID)))
    n_user = len(userid)

    userid_2_index = {v:i for i,v in enumerate(userid)}
    itemid_2_index = {v:i+n_user for i,v in enumerate(itemid)}
    
    data.userID = data.userID.map(userid_2_index)
    data.assessmentItemID = data.assessmentItemID.map(itemid_2_index)
    data.Timestamp = data.Timestamp.apply(lambda x: int(time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timetuple())))
        
    return data[['userID', 'assessmentItemID', 'answerCode', 'Timestamp']]