import wandb
from wandb.lightgbm import wandb_callback
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import wandb
import lightgbm as lgb
import numpy as np
import os
import optuna
from optuna.samplers import TPESampler

def get_lgbm_model(config, train, y_train, test, y_test, FEATS):
    
    lgb_train = lgb.Dataset(train[FEATS], y_train)
    lgb_test = lgb.Dataset(test[FEATS], y_test)

    model = lgb.train(
            {'objective': 'binary'}, 
            lgb_train,
            valid_sets=[lgb_train, lgb_test],
            verbose_eval=config['trainer']['verbos_eval'],
            num_boost_round=config['trainer']['num_boost_round'],
            early_stopping_rounds=config['trainer']['early_stopping_rounds'],
            callbacks=[wandb_callback()]
        )

    if not os.path.exists(config['trainer']['model_dir']):
        os.makedirs(config['trainer']['model_dir'])
    model_path = os.path.join(config['trainer']['model_dir'], "lgbm_model.txt")
    model.save_model(model_path)

    return model

def get_lgbm_optuna(config, train, y_train, test, y_test, FEATS):
    sampler = TPESampler(config['seed'])
    def objective(trial):
        dtrain = lgb.Dataset(train[FEATS], y_train)
        dtest = lgb.Dataset(test[FEATS], y_test)

        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'seed': 42
        }
        model = lgb.train(
            param, 
            dtrain,
            valid_sets=[dtrain, dtest],
            verbose_eval=100,
            num_boost_round=500,
            early_stopping_rounds=100,
        )

        preds = model.predict(test[FEATS])
        acc = accuracy_score(y_test, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(y_test, preds)

        wandb.log({"ACC": acc, "AUC": auc})

        return auc

    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective,  n_trials=100)

    trial = study.best_trial
    trial_params = trial.params
    print('Best Trial: score {},\nparams {}'.format(trial.value, trial_params))

    # 최적의 파라미터로 모델 재학습
    final_lgb_model = lgb.LGBMClassifier(**trial_params)
    final_lgb_model.fit(train[FEATS], y_train)
    return final_lgb_model




"""import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
"""