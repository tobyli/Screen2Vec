import argparse
import json
from torch.utils.data import DataLoader
from UI2Vec import UI2Vec
from prepretrainer import UI2VecTrainer
from prediction import HiddenLabelPredictorModel
from dataset.dataset import RicoDataset, RicoScreen, ScreenDataset
from dataset.vocab import BertScreenVocab
from sentence_transformers import SentenceTransformer
from plotter import plot_loss


parser = argparse.ArgumentParser()

parser.add_argument("-c", "--train_dataset", required=True, type=str, help="dataset to train model")
parser.add_argument("-t", "--test_dataset", type=str, default=None, help="dataset to test model")
parser.add_argument("-o", "--output_path", required=True, type=str, help="where to store model")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="traces in a batch")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-v", "--vocab_path", required=True, type=str, help="path to file with full vocab")
parser.add_argument("-n", "--num_predictors", type=int, default=10, help="number of other labels used to predict one")


args = parser.parse_args()

bert = SentenceTransformer('bert-base-nli-mean-tokens')
with open(args.vocab_path) as f:
    vocab_list = json.load(f, encoding='utf8')

vocab = BertScreenVocab(vocab_list, len(vocab_list), bert)

print("Length of vocab is " + str(len(vocab_list)))
train_dataset_rico = RicoDataset(args.train_dataset)
test_dataset_rico = RicoDataset(args.test_dataset)
train_dataset = ScreenDataset(train_dataset_rico, args.num_predictors)
test_dataset = ScreenDataset(test_dataset_rico, args.num_predictors)

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)

model = UI2Vec(bert)

predictor = HiddenLabelPredictorModel(model, bert, 768, args.num_predictors) 

trainer = UI2VecTrainer(model, predictor, train_data_loader, test_data_loader, vocab, len(vocab_list), 0.01, args.num_predictors, 768)

test_loss_data = []
train_loss_data = []
for epoch in range(args.epochs):
    print(epoch)
    train_loss = trainer.train(epoch)
    print(train_loss)
    train_loss_data.append(train_loss)
    if test_data_loader is not None:
        test_loss = trainer.test(epoch)
        test_loss_data.append(test_loss)
trainer.save(args.epochs, args.output_path)
plot_loss(train_loss_data, test_loss_data)
