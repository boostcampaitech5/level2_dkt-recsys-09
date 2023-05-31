import pandas as pd
import os
import argparse
import sys

sys.path.append('/opt/ml/level2_dkt-recsys-09/DKT')

from data_loader.preprocess_ML import load_data, feature_engineering, custom_train_test_split, categorical_label_encoding, convert_time, scaling
import lightgbm as lgb
import joblib
from utils import read_json, set_seed
import warnings
warnings.filterwarnings(action='ignore')


def main(config):
    # Load Testdata
    print('*'*20 + "Preparing data ..." + '*'*20)
    test_df = load_data(config, config['data_loader']['df_test'])


    # Feature Engineering
    print('*'*17 + "Start Preprocessing ..." + '*'*18)
    test_df["Timestamp"] = test_df["Timestamp"].apply(convert_time)
    if config['data_loader']['feature_engineering']:
        test_df = feature_engineering(os.path.join(config['data_loader']['data_dir'], config['data_loader']['fe_test']), test_df)
        print('*'*17 + "Done feature engineering" + '*'*17)
    else:
        test_df = load_data(config, config['data_loader']['fe_test'])
        print('*'*15 + "LOAD feature engineering data" + '*'*15)
    test_df = categorical_label_encoding(config, test_df, is_train=False)
    #test_df = scaling(args, test_df, is_train=False)
    print('*'*20 + "Done Preprocessing" + '*'*20)

    # Leave Last Interaction Only 
    test_df = test_df[test_df['userID'] != test_df['userID'].shift(-1)]
    test_df.fillna(0, axis=1, inplace=True)
    # Drop Target Feature
    test_df = test_df.drop(['answerCode'], axis=1)


    # Prediction
    print('*'*20 + "Start Predict ..." + '*'*21)
    FEATS = [col for col in test_df.select_dtypes(include=["int", "int8", "int16", "int64", "float", "float16", "float64"]).columns if col not in ['answerCode']]
    if config['trainer']['tuning'] == False:
        model = joblib.load(config['trainer']['model_dir'] + 'lgbm_model.pkl')
    else:
        model = joblib.load(config['trainer']['model_dir'] + 'lgbm_optuna_model.pkl')
    total_preds = model.predict(test_df[FEATS])
    print('*'*20 + "Done Predict" + '*'*26)

    # Save Output
    if config['trainer']['tuning'] == False:
        write_path = os.path.join(config['data_loader']['data_dir'], "lgbm_submission.csv")
    else:
        write_path = os.path.join(config['data_loader']['data_dir'], "lgbm_optuna_submission.csv")
    submission = pd.read_csv(config['data_loader']['data_dir'] + 'sample_submission.csv')
    submission['prediction'] = total_preds
    submission.to_csv(write_path)
    print('*'*25 + "Finish!!" + '*'*25)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="DKT FFM")
    args.add_argument(
        "-c",
        "--config",
        default="config/config_LGBM.json",
        type=str,
        help='config 파일 경로 (default: "./config.json")',
    )
    args = args.parse_args()
    config = read_json(args.config)

    main(config)