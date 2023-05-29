import argparse
import torch
import model.model_GCN as module_arch
from parse_config import ConfigParser
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from data_loader.data_preprocess_HM import Preprocess
from data_loader.data_loaders_GCN import HMDataset


def main(config):
    preprocess = Preprocess(config['data_loader']['args'])
    preprocess.load_test_data("test_data.csv")
    data = preprocess.get_test_data()
    
    test_dataset = HMDataset(data, config['data_loader']['args']['max_seq_len'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['test']['batch_size'],  shuffle=False, collate_fn=collate)

    # build model architecture
    model = config.init_obj('arch', module_arch).to('cuda')
    model.load_state_dict(torch.load(config['test']['model_dir'])['state_dict'])
    model.eval()
    
    with torch.no_grad():
        predicts = list()
        for idx, data in enumerate(test_dataloader):
            input = list(map(lambda t: t.to('cuda'), process_batch(data)))
            output = model(input)[:, -1]
            predicts.extend(output.tolist())
        
    write_path = config['test']['submission_dir']
    submission = pd.read_csv(config['test']['sample_submission_dir'])
    submission['prediction'] = predicts
    submission.to_csv(write_path, index=False)


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def process_batch(batch):

        test, question, tag, correct, mask, user_mean, user_acc, elap_time, recent3_elap_time, assess_ans_mean, prefix = batch

        # change to float
        mask = mask.float()
        correct = correct.float()

        # interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
        interaction = correct + 1  # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1, dims=1)
        interaction_mask = mask.roll(shifts=1, dims=1)
        interaction_mask[:, 0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)

        #  test_id, question_id, tag
        test = ((test + 1) * mask).int()
        question = ((question + 1) * mask).int()
        tag = ((tag + 1) * mask).int()

        return (test, question, tag, correct, mask, interaction, user_mean, user_acc, elap_time, recent3_elap_time, assess_ans_mean, prefix)


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
