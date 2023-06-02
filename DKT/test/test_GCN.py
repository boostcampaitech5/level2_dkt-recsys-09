import argparse
import torch
import sys
import os

#sys.path.append('/opt/ml/level2_dkt-recsys-09/DKT')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import model.model_GCN as module_arch
from parse_config import ConfigParser
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def main(config):
    data = pd.read_csv(config['test']['data_dir']).drop('answerCode', axis=1)
    test_dataset = TensorDataset(torch.LongTensor(data.values))
    test_dataloader = DataLoader(test_dataset, batch_size=config['test']['batch_size'],  shuffle=False)

    if config['fold']:
        predicts_list = list()
        for fold in range(5):
            # build model architecture
            model = config.init_obj('arch', module_arch)
            model_path = config['test']['model_dir']+"{}.pth".format(fold)
            model.load_state_dict(torch.load(model_path)['state_dict'])
            model.eval()
            
            predict = list()
            for idx, data in enumerate(test_dataloader):
                predict.extend(model(data[0]).tolist())
            predicts_list.append(predict)
        predicts = np.mean(predicts_list, axis=0)
    else:
        # build model architecture
        model = config.init_obj('arch', module_arch)
        model_path = config['test']['model_dir']+"0.pth"
        model.load_state_dict(torch.load(model_path)['state_dict'])
        model.eval()
    
        predicts = list()
        for idx, data in enumerate(test_dataloader):
            predicts.extend(model(data[0]).tolist())
        
    dir_path = "./submission"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    write_path = config['test']['submission_dir']
    submission = pd.read_csv(config['test']['sample_submission_dir'])
    submission['prediction'] = predicts
    submission.to_csv(write_path, index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config/config_ultraGCN.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
