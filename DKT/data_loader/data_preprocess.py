import pandas as pd

def ultragcn_preprocess(train, test):
    
    # 한 유저가 같은 문제를 여러 번 푼 경우 마지막 성적만을 반영
    data = pd.concat(train, test).drop_duplicates(subset = ["userID", "assessmentItemID"],
                                                  keep = "last", inplace = True)
    
    # answerCode가 -1인 항목 제거 (평가 항목 제거)
    data = data[data.answerCode >= -0]
    
    return data.drop('Timestamp', axis=1)
    
    