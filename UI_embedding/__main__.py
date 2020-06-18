import argparse
import json
from torch.utils.data import DataLoader
from .model import UI2Vec
from .trainer import UI2VecTrainer
from .dataset import RicoDataset, RicoTrace, RicoScreen, ScreenDataset

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="dataset to train model")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="dataset to test model")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="where to store model")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="traces in a batch")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="path to file with full vocab")
    parser.add_argument("-n", "--num_predictors", type=int, default=10, help="number of other labels used to predict one")

    
    args = parser.parse_args()
    

    with open(args.vocab_path) as f:
        vocab = json.load(f)
    train_dataset_rico = RicoDataset(args.train_dataset)
    test_dataset_rico = RicoDataset(args.test_dataset)
    train_dataset = ScreenDataset(train_dataset_rico)
    test_dataset = ScreenDataset(test_dataset_rico)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = UI2Vec()

    trainer = UI2VecTrainer(model, train_data_loader, test_data_loader, len(vocab), 0.01, args.num_predictors, 768)

    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)
        if test_data_loader is not None:
            trainer.test(epoch)