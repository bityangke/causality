import torch
import argparse
import logging
from os import path
import os
import sys

# CHANGE THIS TO ORIG SPLIT IF NEEDED
from diff_data_loader import COPADataset
from model import COPAModel
import train


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


def get_model_name(hp):
    # Only include important options in hash computation
    hp_dict = vars(hp)
    model_name = ("copa_" + hp_dict['model'] + "_" + hp_dict['model_size']
                  + '_' + hp_dict['pool_method'])
    return model_name


def main():
    hp = parse_args()

    # Setup model directories
    model_name = get_model_name(hp)
    model_path = path.join(hp.model_dir, model_name)
    if not path.exists(model_path):
        os.makedirs(model_path)

    # Set random seed
    torch.manual_seed(hp.seed)

    # Initialize the model
    model = COPAModel(**vars(hp)).cuda()
    sys.stdout.flush()

    # Load data
    logging.info("Loading data")
    train_iter, val_iter, test_iter = COPADataset.iters(
        hp.data_dir, model.encoder, batch_size=hp.batch_size)
    logging.info("Data loaded")

    optimizer = torch.optim.Adam(model.get_other_params(), lr=hp.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.5, verbose=True)

    val_f1, test_f1 = train.train(
        model, train_iter, val_iter, test_iter, optimizer, scheduler,
        max_epochs=hp.n_epochs)

    perf_dir = path.join(hp.model_dir, "perf")
    if not path.exists(perf_dir):
        os.makedirs(perf_dir)
    perf_file = path.join(perf_dir, model_name + ".txt")
    with open(perf_file, "w") as f:
        f.write("%s\t%.4f\n" % ("Valid", val_f1))
        f.write("%s\t%.4f\n" % ("Test", test_f1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data_dir", type=str,
        default="/home/shtoshni/Research/causality/tasks/copa/data/diff_split")
    parser.add_argument(
        "-model_dir", type=str,
        default="/home/shtoshni/Research/causality/tasks/copa/checkpoints")
    parser.add_argument("-batch_size", type=int, default=2)
    parser.add_argument("-eval_batch_size", type=int, default=32)
    parser.add_argument("-n_epochs", type=int, default=20)
    parser.add_argument("-lr", type=float, default=5e-4)
    parser.add_argument("-model", type=str, default="bert")
    parser.add_argument("-model_size", type=str, default="base")
    parser.add_argument("-emb_size", type=int, default=10,
                        help="Embedding size of relationship (cause/effect).")
    parser.add_argument("-pool_method", default="avg", type=str)
    parser.add_argument("-seed", type=int, default=0, help="Random seed")
    parser.add_argument("-eval", default=False, action="store_true")

    hp = parser.parse_args()
    return hp


if __name__ == "__main__":
    main()
