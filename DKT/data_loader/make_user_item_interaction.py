import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

def get_count(df, id):
    count_id = df[[id, 'rating']].groupby(id, as_index=False)
    return count_id.size()

def filter(df, min_user_count, min_item_count):
    item_count = get_count(df, 'iid')
    user_count = get_count(df, 'uid')

    return df, user_count, item_count


def numerize(df, user2id):
    
    uid = list(map(lambda x: user2id[x], df['uid']))
    df['uid_new'] = uid
    
    le1 = LabelEncoder()
    id_lists = df["iid"].unique().tolist() + ["unknown"]
    le1.fit(id_lists)
    df['iid_new'] = df['iid']
    iid_new = le1.transform(df['iid_new'].astype(str))
    df['iid_new'] = iid_new
    
    le2 = LabelEncoder()
    tag_lists = df["KnowledgeTag"].unique().tolist() + ["unknown"]
    le2.fit(tag_lists)
    df['KnowledgeTag_new'] = df['KnowledgeTag']
    df['KnowledgeTag_new'] = le2.transform(df['KnowledgeTag_new'].astype(str))
    
    return df

def __make_user_item_interaction(config, train_df, test_df):
    print('data preprocessing...')

    df = pd.concat([train_df, test_df])

    df = df.sort_values(by=["userID", "Timestamp"], axis=0) 

    df.rename(columns={'userID': 'uid', 'assessmentItemID': 'iid', 'answerCode': 'rating'}, inplace=True) # userID를 user로 assessmentID를 item으로 answerCode를 rating으로 생각하기 위해 컬럼명 변경 

    df, user_count, item_count = filter(df, min_user_count=20, min_item_count=20) # 최소 사용자 수와 최소 아이템 수를 충족시키지 않은 행을 제거 후 df, 사용자 수, 아이템 수를 반환
                                                                                  # 일단은 20으로 설정

    sparsity = float(df.shape[0]) / user_count.shape[0] / item_count.shape[0]
    print('num_user: %d, num_item: %d, num_interaction: %d, sparsity: %.4f%%' % (user_count.shape[0], item_count.shape[0], df.shape[0], sparsity * 100))

    unique_uid = user_count.index
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
    all_df = numerize(df, user2id)

    print('data splitting...')

    all_df_sorted = all_df.sort_values(by=['uid_new', 'Timestamp', 'iid_new'])

    users = np.array(all_df_sorted['uid_new'], dtype=np.int32)
    items = np.array(all_df_sorted['iid_new'], dtype=np.int32)

    all_data = defaultdict(list) # 딕셔너리에 새로운 원소를 쉽게 추가하기 위해 defaultdict로 바꿈
    for n in range(len(users)):
        all_data[users[n]].append(items[n]) # user-item interaction dict

    train_dict = dict()

    for u in all_data:
        train_dict[u] = all_data[u][:-2]


    print('preprocessed data save')
    
    data_dir = config['data_loader']['args']['data_dir']
    np.save(os.path.join(data_dir, 'preprocessed_data'), np.array([train_dict, max(users) + 1, max(items) + 1]))
    tag_df_sorted = all_df.sort_values(by=['KnowledgeTag_new', 'iid_new'])
    grouped_tag = tag_df_sorted.groupby('KnowledgeTag_new').apply(lambda r: list(set(r['iid_new'].values)))
    rel_dict = grouped_tag.to_dict()
    np.save(os.path.join(data_dir, 'preprocessed_data_rel'), np.array([rel_dict]))
    
    print('Making user-item interaction dict is done.')

    return train_dict, rel_dict