import argparse
import collections
import torch
import numpy as np
import sys
import os

#sys.path.append('/opt/ml/level2_dkt-recsys-09/DKT')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data_loader.data_loaders_GCN import UltraGCNDataLoader
import model.loss_GCN as module_loss
import model.metric_GCN as module_metric
import model.model_GCN as module_arch
from parse_config import ConfigParser
from trainer.trainer_GCN import Trainer
from utils import prepare_device
import wandb


import data_loader.data_loaders_GCN as module_data
os.environ['wandb mode'] = 'offline'

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    wandb.login()
    logger = config.get_logger('train')

    wandb.init(project=config['name'], config=config, entity="ffm")

    if config['fold']:
        for fold in range(5):
            print(
                f"-------------------------START FOLD {fold + 1} TRAINING---------------------------"
            )
            print(
                f"-------------------------START FOLD {fold + 1} MODEL LOADING----------------------"
            )

            data_loader = UltraGCNDataLoader(data_dir=config['data_loader']['args']['data_dir'], batch_size=config['data_loader']['args']['batch_size'],
                                            shuffle=config['data_loader']['args']['shuffle'], num_workers=config['data_loader']['args']['num_workers'],
                                            validation_split=config['data_loader']['args']['validation_split'], random_seed=config['data_loader']['args']['random_seed'],
                                            fold=fold)
            valid_data_loader = data_loader.split_validation()

            model = config.init_obj('arch', module_arch)
            logger.info(model)

            # prepare for (multi-device) GPU training
            device, device_ids = prepare_device(config['n_gpu'])
            model = model.to(device)
            if len(device_ids) > 1:
                model = torch.nn.DataParallel(model, device_ids=device_ids)

            # get function handles of loss and metrics
            criterion = getattr(module_loss, config['loss'])
            metrics = [getattr(module_metric, met) for met in config['metrics']]

            # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

            print(
                f"-------------------------DONE FOLD {fold + 1} MODEL LOADING-----------------------"
            )

            trainer = Trainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            data_loader=data_loader,
                            valid_data_loader=valid_data_loader,
                            lr_scheduler=lr_scheduler,
                            fold=fold)

            trainer.train()
            print(
                f"---------------------------DONE FOLD {fold + 1} TRAINING--------------------------"
            )
    else:
        # setup data_loader instances
        data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = data_loader.split_validation()

        wandb.init(project=config['name'], config=config, entity="ffm")
        # build model architecture, then print to console
        model = config.init_obj('arch', module_arch)
        logger.info(model)

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)

        trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config/config_ultraGCN.json', type=str,
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
