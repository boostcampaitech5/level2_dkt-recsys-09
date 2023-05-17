import pandas as pd
import os
from model.preprocess_ML import load_data, feature_engineering, custom_train_test_split, categorical_label_encoding, convert_time, scaling
import lightgbm as lgb
import joblib
from args import parse_args_test
import warnings
warnings.filterwarnings(action='ignore')


if __name__ == "__main__":
    args = parse_args_test()


    # Load Testdata
    print('*'*20 + "Preparing data ..." + '*'*20)
    test_df = load_data(args)


    # Feature Engineering
    print('*'*17 + "Start Preprocessing ..." + '*'*18)
    test_df["Timestamp"] = test_df["Timestamp"].apply(convert_time)
    test_df = feature_engineering(test_df)
    test_df = categorical_label_encoding(args, test_df, is_train=False)
    #test_df = scaling(args, test_df, is_train=False)
    print('*'*20 + "Done Preprocessing" + '*'*20)

    # Leave Last Interaction Only 
    test_df = test_df[test_df['userID'] != test_df['userID'].shift(-1)]
    test_df.fillna(0, axis=1, inplace=True)


    # Drop Target Feature
    test_df = test_df.drop(['answerCode'], axis=1)


    # Prediction
    print('*'*20 + "Start Predict ..." + '*'*20)
    FEATS = [col for col in test_df.select_dtypes(include=["int", "int8", "int16", "int64", "float", "float16", "float64"]).columns if col not in ['answerCode']]
    #model = lgb.Booster(model_file=os.path.join(args.model_dir, "lgbm_model.txt")) # Load saved model
    model = joblib.load(args.model_dir + 'lgbm_optuna_model.pkl')
    total_preds = model.predict(test_df[FEATS])
    print('*'*20 + "Done Predict" + '*'*25)

    # Save Output
    write_path = os.path.join(args.data_dir, "lgbm_optuna_submission.csv")
    submission = pd.read_csv(args.data_dir + 'sample_submission.csv')
    submission['prediction'] = total_preds
    submission.to_csv(write_path)
    print('*'*20 + "Finish!!" + '*'*20)

























"""import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
"""