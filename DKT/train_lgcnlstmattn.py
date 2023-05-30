import os
import numpy as np
import torch
import wandb
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds, get_adj_matrix, get_adj_matrix_wo_rel, get_adj_matrix_wo_normarlize
import random


def main(args):
    wandb.login()

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    
    [train_dict, num_user, num_item] = np.load('/opt/ml/input/data/preprocessed_data.npy', allow_pickle=True)
    rel_dict = np.load('/opt/ml/input/data/preprocessed_data_rel.npy', allow_pickle=True)[0]
    print('num_user:%d, num_item:%d' % (num_user, num_item))
    args.gcn_n_items = num_item
    
    train_dict_len = [len(train_dict[u]) for u in train_dict]
    print('max len: %d, min len:%d, avg len:%.2f' % (np.max(train_dict_len), np.min(train_dict_len), np.mean(train_dict_len)))
    
    
    # adj_matrix_wo_normarlize = get_adj_matrix_wo_normarlize(train_dict, num_item, args.max_seq_len)
    adj_matrix = get_adj_matrix(train_dict, rel_dict, num_item, args.alpha, args.beta, args.max_seq_len)
    
    
    print('Model preparing...')
    
    preprocess = Preprocess(args=args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    train_data, valid_data = preprocess.split_data(train_data)
    
    name_dict = {
        'model': args.model,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'max_seq_len': args.max_seq_len,
        'hidden_dim': args.hidden_dim,
    }
    
    name = ''
    for key, value in name_dict.items():
        name += f'{key}_{value}, '
        
    wandb.init(project="LGCNtrans", config=vars(args), name=name, entity="ffm")
    model = trainer.get_model(args, adj_matrix).to(args.device)
    # trainer.run(args, train_data, valid_data, model)
    trainer.run_with_vaild_loss(args, train_data, valid_data, model)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)