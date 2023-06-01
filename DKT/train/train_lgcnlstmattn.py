import os
import numpy as np
import pandas as pd
import torch
import wandb
import sys

sys.path.append('/opt/ml/level2_dkt-recsys-09/DKT')

from trainer import trainer_lgcnlstmattn
from data_loader.make_user_item_interaction import __make_user_item_interaction
from data_loader.dataloader_lgcnlstmattn import Preprocess
from utils.util_lgcnlstmattn import setSeeds, get_adj_matrix
import random
from parse_config import ConfigParser
import argparse
import collections
import os

def main(args):
    #wandb.login()

    setSeeds(args['seed'])
    args['model']['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists('/opt/ml/input/data/preprocessed_data.npy'):
        train_df = pd.read_csv("/opt/ml/input/data/train_data.csv")
        test_df = pd.read_csv("/opt/ml/input/data/test_data.csv")
        __make_user_item_interaction(args, train_df, test_df)
        
    [train_dict, num_user, num_item] = np.load('/opt/ml/input/data/preprocessed_data.npy', allow_pickle=True)
    rel_dict = np.load('/opt/ml/input/data/preprocessed_data_rel.npy', allow_pickle=True)[0]    
    
    print('num_user:%d, num_item:%d' % (num_user, num_item))
    args['model']['gcn_n_items'] = num_item
    
    train_dict_len = [len(train_dict[u]) for u in train_dict]
    print('max len: %d, min len:%d, avg len:%.2f' % (np.max(train_dict_len), np.min(train_dict_len), np.mean(train_dict_len)))
    
    model_params = args['model']
    # adj_matrix_wo_normarlize = get_adj_matrix_wo_normarlize(train_dict, num_item, args.max_seq_len)
    adj_matrix = get_adj_matrix(train_dict, rel_dict, num_item, model_params['alpha'], model_params['beta'], model_params['max_seq_len'])
    print('Model preparing...')
    
    preprocess = Preprocess(args)
    preprocess.load_train_data('train_data.csv')
    train_data = preprocess.get_train_data()

    train_data, valid_data = preprocess.split_data(train_data)
    
    
    trainer_params = args['trainer']
    
    name_dict = {
        'model': model_params['name'],
        'n_epochs': trainer_params['n_epochs'],
        'batch_size': trainer_params['batch_size'],
        'lr': trainer_params['lr'],
        'max_seq_len': model_params['max_seq_len'],
        'hidden_dim': model_params['hidden_dim'],
    }
    
    name = ''
    for key, value in name_dict.items():
        name += f'{key}_{value}, '
        
    wandb.init(project="LGCNtrans", config=vars(args), name=name, entity="ffm")
    
    model = trainer_lgcnlstmattn.get_model(adj_matrix, **model_params).to(args['model']['device'])
    trainer_lgcnlstmattn.run_with_vaild_loss(args, train_data, valid_data, model)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="./config/config_lgcnlstmattn.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)