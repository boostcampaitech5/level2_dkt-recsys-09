import os

import torch
from args import parse_args
from trainer import trainer_LQ
from data_loader.data_preprocess_LQ import Preprocess


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()

    trainer_LQ.inference(args, test_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)