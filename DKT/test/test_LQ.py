import os
import torch
import sys

#sys.path.append('/opt/ml/level2_dkt-recsys-09/DKT')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from args_LQ import parse_args
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