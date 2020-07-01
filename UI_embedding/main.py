import argparse
import json
import tqdm
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
parser.add_argument("-t", "--test_dataset", required=False, type=str, default=None, help="dataset to test model")
parser.add_argument("-o", "--output_path", required=True, type=str, help="where to store model")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="traces in a batch")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-v", "--vocab_path", required=True, type=str, help="path to file with full vocab")
parser.add_argument("-m", "--embedding_path",  type=str, default=None, help="path to file with precomputed vocab embeddings")
parser.add_argument("-n", "--num_predictors", type=int, default=10, help="number of other labels used to predict one")
parser.add_argument("-l", "--loss", type=int, default=1, help="1 to use cosine embedding loss, 0 to use softmax dot product")
parser.add_argument("-r", "--rate", type=float, default=0.001, help="learning rate")


args = parser.parse_args()

bert = SentenceTransformer('bert-base-nli-mean-tokens')
with open(args.vocab_path) as f:
    vocab_list = json.load(f, encoding='utf-8')

vocab = BertScreenVocab(vocab_list, len(vocab_list), bert, args.embedding_path)

print("Length of vocab is " + str(len(vocab_list)))
train_dataset_rico = RicoDataset(args.train_dataset)
train_dataset = ScreenDataset(train_dataset_rico, args.num_predictors)
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size)

if args.test_dataset:
    test_dataset_rico = RicoDataset(args.test_dataset)
    test_dataset = ScreenDataset(test_dataset_rico, args.num_predictors)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)
else: 
    test_data_loader = None

predictor = HiddenLabelPredictorModel(bert, 768, args.num_predictors) 

trainer = UI2VecTrainer(predictor, train_data_loader, test_data_loader, vocab, len(vocab_list), args.rate, args.num_predictors, args.loss, 768)

test_loss_data = []
train_loss_data = []
for epoch in tqdm.tqdm(range(args.epochs)):
    print(epoch)
    train_loss = trainer.train(epoch)
    print(train_loss)
    train_loss_data.append(train_loss)
    if test_data_loader is not None:
        test_loss = trainer.test(epoch)
        test_loss_data.append(test_loss)
    if (epoch%20)==0:
        trainer.save(epoch, args.output_path)
trainer.save(args.epochs, args.output_path)
plot_loss(train_loss_data, test_loss_data)
