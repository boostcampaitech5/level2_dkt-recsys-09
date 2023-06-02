from torch.utils.data import DataLoader, Dataset
from base import BaseDataLoader
import pandas as pd
import os
from .data_preprocess_GCN import ultragcn_preprocess
from .data_preprocess_HM import Preprocess
import torch
import numpy as np

class UltraGCNDataset(Dataset):
    def __init__(self, data_dir):
        
        if not os.path.exists(os.path.join(data_dir, "data.csv")):
            self.train = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
            self.test = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
            ultragcn_preprocess(self.train, self.test)

        self.data = pd.read_csv(os.path.join(data_dir, "data.csv"))
        self.X = self.data.drop('answerCode', axis=1)
        self.y = self.data.answerCode
        
    def __getitem__(self, index):
        return self.X.loc[index].values, self.y.loc[index]
    
    def __len__(self):
        return len(self.data)      


class UltraGCNDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=False, num_workers=1, validation_split=0.0, random_seed=42, fold=0):
        
        self.data_dir = data_dir
        self.random_seed = random_seed
        self.dataset = UltraGCNDataset(data_dir)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, fold)
        
class HMDataset(Dataset):
    def __init__(self, data, max_seq_len):
        self.data = data
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        # cate
        test, question, tag, correct = row[0], row[1], row[2], row[3]
        
        # cont
        user_mean, user_acc, elap_time, recent3_elap_time = np.log1p(row[4]), np.log1p(row[5]), np.log1p(row[6]), np.log1p(row[7])
        assess_ans_mean, prefix = np.log1p(row[8]), np.log1p(row[9])

        cate_cols = [test, question, tag, correct]
        cont_columns = [user_mean, user_acc, elap_time, recent3_elap_time, assess_ans_mean, prefix]
        total_cols = cate_cols + cont_columns

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.max_seq_len:
            for i, col in enumerate(total_cols):
                total_cols[i] = col[-self.max_seq_len :]
            mask = np.ones(self.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        total_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(total_cols):
            total_cols[i] = torch.tensor(col)

        return total_cols

    def __len__(self):
        return len(self.data)   
    

class HMDataLoader(BaseDataLoader):
    def __init__(self, **args):
        self.preprocess = Preprocess(args)
        self.preprocess.load_train_data("train_data.csv")
        self.data = self.preprocess.get_train_data()
        self.data = self.preprocess.data_augmentation(self.data)
        self.dataset = HMDataset(self.data, args['max_seq_len']) 
        
        super().__init__(self.dataset, args['batch_size'], args['shuffle'], args['validation_split'], args['num_workers'], collate_fn=self.collate, fold=0)

    def collate(self, batch):
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
