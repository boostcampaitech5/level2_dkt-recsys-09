import os
import random
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .feature_engine import fe
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

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args['asset_dir'], name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]
        

        if not os.path.exists(self.args['asset_dir']):
            os.makedirs(self.args['asset_dir'])
        
        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args['asset_dir'], col + "_classes.npy")
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
            
        if os.path.exists(f"/opt/ml/input/data/{csv}_featured.csv"):
            df = pd.read_csv(f"/opt/ml/input/data/{csv}_featured.csv")
        else:
            df = fe(df)
            df.to_csv(f"/opt/ml/input/data/{csv}_featured.csv")
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args['data_dir'], file_name)
        df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])  # , nrows=100000) 
        df = self.__feature_engineering(df, is_train)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args['n_questions'] = len(
            np.load(os.path.join(self.args['asset_dir'], "assessmentItemID_classes.npy"))
        )
        self.args['n_test'] = len(
            np.load(os.path.join(self.args['asset_dir'], "testId_classes.npy"))
        )
        self.args['n_tag'] = len(
            np.load(os.path.join(self.args['asset_dir'], "KnowledgeTag_classes.npy"))
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