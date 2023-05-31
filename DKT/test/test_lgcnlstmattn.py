import os
import torch
import sys

sys.path.append('/opt/ml/level2_dkt-recsys-09/DKT')

from trainer import trainer_lgcnlstmattn
from data_loader.dataloader_lgcnlstmattn import Preprocess
from utils.util_lgcnlstmattn import get_adj_matrix
import numpy as np
from args import parse_args
import argparse
from parse_config import ConfigParser

def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    
    
    [train_dict, num_user, num_item] = np.load('/opt/ml/input/data/preprocessed_data.npy', allow_pickle=True)
    rel_dict = np.load('/opt/ml/input/data/preprocessed_data_rel.npy', allow_pickle=True)[0]
    print('num_user:%d, num_item:%d' % (num_user, num_item))
    args.gcn_n_items = num_item
    
    train_dict_len = [len(train_dict[u]) for u in train_dict]
    print('max len: %d, min len:%d, avg len:%.2f' % (np.max(train_dict_len), np.min(train_dict_len), np.mean(train_dict_len)))
    
    
    # adj_matrix_wo_normarlize = get_adj_matrix_wo_normarlize(train_dict, num_item, args.max_seq_len)
    adj_matrix = get_adj_matrix(train_dict, rel_dict, num_item, args.alpha, args.beta, args.max_seq_len)
    
    
    test_data = preprocess.get_test_data()
    # model = trainer.get_model(args).to(args.device)
    model = trainer_lgcnlstmattn.load_model(args, adj_matrix).to(args.device)
    trainer_lgcnlstmattn.inference(args, test_data, model)


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