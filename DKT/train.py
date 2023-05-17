import os
import torch
import wandb
import lightgbm as lgb
from matplotlib import pyplot as plt

from args import parse_args_train
from model.preprocess_ML import load_data, feature_engineering, custom_train_test_split, categorical_label_encoding, convert_time
from trainer.trainer_ML import train_model


if __name__ == "__main__":
    # init
    wandb.login()
    args = parse_args_train()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Load
    print('*'*20 + "Preparing data ..." + '*'*20)
    df = load_data(args)

    # Preprocessing
    print('*'*17 + "Start Preprocessing ..." + '*'*18)
    df["Timestamp"] = df["Timestamp"].apply(convert_time)
    df = feature_engineering(df)
    df = categorical_label_encoding(args, df, is_train=True) # LGBM을 위한 FE
    
    train, test = custom_train_test_split(args, df)
    print('*'*20 + "Done Preprocessing" + '*'*20)

    # Make new_wandb project
    wandb.init(project="dkt", config=vars(args))

    
    # Train model
    print('*'*20 + "Start Training ..." + '*'*20)
    FEATS = [col for col in df.select_dtypes(include=["int", "int8", "int16", "int64", "float", "float16", "float64"]).columns if col not in ['answerCode']]
    trained_model = train_model(args, train, test, FEATS)
    print('*'*20 + "Done Training" + '*'*25)


    # Save a feature importance
    x = lgb.plot_importance(trained_model)
    if not os.path.exists(args.pic_dir):
        os.makedirs(args.pic_dir)
    plt.savefig(os.path.join(args.pic_dir, 'lgbm_feature_importance.png'))
    
    print('*'*25 + "Finish!!" + '*'*25)


