import argparse
import torch
import model.model as module_arch
from parse_config import ConfigParser
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


def main(config):
    data = pd.read_csv(config['test']['data_dir']).drop('answerCode', axis=1)
    test_dataset = TensorDataset(torch.LongTensor(data.values))
    test_dataloader = DataLoader(test_dataset, batch_size=config['test']['batch_size'],  shuffle=False)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    model.load_state_dict(torch.load(config['test']['model_dir'])['state_dict'])
    model.eval()
    
    predicts = list()
    for idx, data in enumerate(test_dataloader):
        predicts.extend(model(data[0]).tolist())
        
    write_path = config['test']['submission_dir']
    submission = pd.read_csv(config['test']['sample_submission_dir'])
    submission['prediction'] = predicts
    submission.to_csv(write_path, index=False)


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
