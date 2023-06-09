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
from .feature_engine_lgcnlstmattn import fe
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.8, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """

        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args['data_loader']['args']['asset_dir'], name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]
        

        if not os.path.exists(self.args['data_loader']['args']['asset_dir']):
            os.makedirs(self.args['data_loader']['args']['asset_dir'])
        
        
        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args['data_loader']['args']['asset_dir'], col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s):
            s = str(s)
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __feature_engineering(self, df, is_train):
        
        csv = 'train' if is_train else 'test'
            
        if os.path.exists(f"/opt/ml/input/data/fe_{csv}_data.csv"):
            df = pd.read_csv(f"/opt/ml/input/data/fe_{csv}_data.csv")
        else:
            df = fe(df)
            df.to_csv(f"/opt/ml/input/data/fe_{csv}_data.csv")
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args['data_loader']['args']['data_dir'], file_name)
        df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])  # , nrows=100000)
        df = self.__feature_engineering(df, is_train)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args['model']['n_questions'] = len(
            np.load(os.path.join(self.args['data_loader']['args']['asset_dir'], "assessmentItemID_classes.npy"))
        )
        self.args['model']['n_test'] = len(
            np.load(os.path.join(self.args['data_loader']['args']['asset_dir'], "testId_classes.npy"))
        )
        self.args['model']['n_tag'] = len(
            np.load(os.path.join(self.args['data_loader']['args']['asset_dir'], "KnowledgeTag_classes.npy"))
        )

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        cat_columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        cont_columns = ['user_mean', 'user_acc', 'elap_time', 'recent3_elap_time', 'elo_prob', 'assess_ans_mean', 'prefix']
        
        columns = cat_columns + cont_columns
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                    r["user_mean"].values,
                    r["user_acc"].values,
                    r["elap_time"].values,
                    r["recent3_elap_time"].values,
                    r["elo_prob"].values,
                    r["assess_ans_mean"].values,
                    r["prefix"].values,
                )
            )
        )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)

    
    
class GESDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        # cate
        test, question, tag, correct = row[0], row[1], row[2], row[3]
        
        # cont
        user_mean, user_acc, elap_time, recent3_elap_time = np.log1p(row[4]), np.log1p(row[5]), np.log1p(row[6]), np.log1p(row[7])
        elo_prob, assess_ans_mean, prefix =  np.log1p(row[8]), np.log1p(row[9]), np.log1p(row[10])

        cate_cols = [test, question, tag, correct]
        cont_columns = [user_mean, user_acc, elap_time, recent3_elap_time, elo_prob, assess_ans_mean, prefix]
        total_cols = cate_cols + cont_columns

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args['max_seq_len']:
            for i, col in enumerate(total_cols):
                total_cols[i] = col[-self.args['max_seq_len'] :]
            mask = np.ones(self.args['max_seq_len'], dtype=np.int16)
        else:
            mask = np.zeros(self.args['max_seq_len'], dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        total_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(total_cols):
            total_cols[i] = torch.tensor(col)

        return total_cols

    def __len__(self):
        return len(self.data)   


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_GES_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = GESDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args['num_workers'],
            shuffle=True,
            batch_size=args['batch_size'],
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = GESDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args['num_workers'],
            shuffle=False,
            batch_size=args['batch_size'],
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    return train_loader, valid_loader