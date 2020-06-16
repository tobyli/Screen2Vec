import argparse
from torch.utils.data import DataLoader
from .model import Screen2Vec
from .trainer import Screen2VecTrainer
from .dataset import RicoDataset, RicoTrace, RicoScreen


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="dataset to train model")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="dataset to test model")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="where to store model")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="traces in a batch")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")


    args = parser.parse_args()

    train_dataset = RicoDataset(args.train_dataset)
    test_dataset = RicoDataset(args.test_dataset)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = Screen2Vec()

    trainer = Screen2VecTrainer(model)

    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)
        if test_data_loader is not None:
            trainer.test(epoch)
