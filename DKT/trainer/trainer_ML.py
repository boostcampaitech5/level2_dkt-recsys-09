import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
from wandb.lightgbm import wandb_callback
from data_loader.preprocess_ML import scaling
import wandb
from model.model_ML import get_lgbm_model, get_lgbm_optuna
import os
import joblib

import optuna
from optuna.samplers import TPESampler

def train_model(config, train, test, FEATS):
    wandb.init(project="DKT_LGBM", entity="ffm")
    wandb.config.update(config)

    # split X, y
    y_train = train['answerCode']
    train = train.drop(['answerCode'], axis=1)

    y_test = test['answerCode']
    test = test.drop(['answerCode'], axis=1)

    # get a model and train
    if config['trainer']['tuning'] == False:
        model = get_lgbm_model(config, train, y_train, test, y_test, FEATS) # base model
    else:
        model = get_lgbm_optuna(config, train, y_train, test, y_test, FEATS) # tunning model
    

    preds = model.predict(test[FEATS])
    
    acc = accuracy_score(y_test, np.where(preds >= config['trainer']['threshold'], 1, 0))
    auc = roc_auc_score(y_test, preds)

    print(f'VALID AUC : {auc} ACC : {acc}\n')
    wandb.log({
        "valid auc": auc,
        "valid acc": acc})
    
    # 모델 저장
    if not os.path.exists(config['trainer']['model_dir']):
        os.makedirs(config['trainer']['model_dir'])
    if config['trainer']['tuning'] == False:
        model_path = os.path.join(config['trainer']['model_dir'], "lgbm_model.pkl") # base model
    else:
        model_path = os.path.join(config['trainer']['model_dir'], "lgbm_optuna_model.pkl") # tunning model
    joblib.dump(model, model_path)
    return model