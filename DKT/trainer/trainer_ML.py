import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
from wandb.lightgbm import wandb_callback
from model.preprocess_ML import scaling
import wandb
from model.model import get_lgbm_model
import os
import joblib

import optuna
from optuna.samplers import TPESampler

def train_model(args, train, test, FEATS):
    wandb.init(project="DKT_LGBM", entity="ffm")
    wandb.config.update(args)

    # split X, y
    y_train = train['answerCode']
    train = train.drop(['answerCode'], axis=1)

    y_test = test['answerCode']
    test = test.drop(['answerCode'], axis=1)

    # scaling some features
    #train = scaling(args, train, is_train=True)
    #test = scaling(args, test, is_train=False)

    # get a model and train
    #model = get_lgbm_model(args, train, y_train, test, y_test, FEATS)
    sampler = TPESampler(args.seed)
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

    print('*'*50)
    """# Predict valid data
    final_lgb_model = lgb.train(
            trial_params, 
            train[FEATS],
            valid_sets=[train[FEATS], y_train],
            verbose_eval=False,
            num_boost_round=500,
            early_stopping_rounds=100,
        )"""
    # 최적의 파라미터로 모델 재학습
    final_lgb_model = lgb.LGBMClassifier(**trial_params)
    final_lgb_model.fit(train[FEATS], y_train)
    #final_lgb_model = lgb.LGBMRegressor(**trial_params)
    #final_lgb_model.fit(train[FEATS], y_train)

    ########################################################################
    preds = final_lgb_model.predict(test[FEATS])
    
    #preds = model.predict(test[FEATS])
    acc = accuracy_score(y_test, np.where(preds >= args.threshold, 1, 0))
    auc = roc_auc_score(y_test, preds)

    print(f'VALID AUC : {auc} ACC : {acc}\n')
    wandb.log({
        "valid auc": auc,
        "valid acc": acc})
    
    print('*'*50)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model_path = os.path.join(args.model_dir, "lgbm_optuna_model.pkl")
    #final_lgb_model.save_model(model_path)
    # 모델 저장
    joblib.dump(final_lgb_model, model_path)
    return final_lgb_model