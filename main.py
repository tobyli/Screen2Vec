import argparse
import torch
from torch.utils.data import DataLoader
from Screen2Vec import Screen2Vec
from pretrainer import Screen2VecTrainer
from dataset.dataset import RicoDataset, RicoTrace, RicoScreen
from sentence_transformers import SentenceTransformer
from UI_embedding.prediction import HiddenLabelPredictorModel


def pad_collate(batch):
    UIs = [trace[0] for trace in batch]
    descr = [trace[1] for trace in batch]

    trace_screen_lengths = []
    for trace_idx in range(len(UIs)):
        screen_lengths = [len(screen) for screen in UIs[trace_idx]]
        trace_screen_lengths.append(screen_lengths)
        UIs[trace_idx] = torch.nn.utils.rnn.pad_sequence(UIs[trace_idx])
        UIs[trace_idx].transpose(0,1)
    UIs = torch.nn.utils.rnn.pad_sequence(UIs)
    UIs.transpose(1,2)
    return UIs, descr, trace_screen_lengths

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--train_dataset", required=True, type=str, help="dataset to train model")
parser.add_argument("-t", "--test_dataset", required=False, type=str, default=None, help="dataset to test model")
parser.add_argument("-o", "--output_path", required=True, type=str, help="where to store model")
parser.add_argument("-m", "--model", required=True, type=str, help="path where pretrained part was stored")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="traces in a batch")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-n", "--num_predictors", type=int, default=10, help="number of other labels used to predict one")
parser.add_argument("-l", "--loss", type=int, default=1, help="1 to use cosine embedding loss, 0 to use softmax dot product")
parser.add_argument("-r", "--rate", type=float, default=0.001, help="learning rate")




args = parser.parse_args()

bert = SentenceTransformer('bert-base-nli-mean-tokens')
bert_size = 768

loaded_model = HiddenLabelPredictorModel(bert, bert_size, args.num_predictors)
loaded_model.load_state_dict(torch.load(args.model))

train_dataset = RicoDataset(loaded_model.model, args.train_dataset, args.num_predictors)
test_dataset = RicoDataset(loaded_model.model, args.test_dataset, args.num_predictors)

train_data_loader = DataLoader(train_dataset, collate_fn=pad_collate, batch_size=args.batch_size)
test_data_loader = DataLoader(test_dataset, collate_fn=pad_collate, batch_size=args.batch_size)

model = Screen2Vec(loaded_model.model, bert, bert_size)

trainer = Screen2VecTrainer(model, train_data_loader, test_data_loader)

for epoch in range(args.epochs):
    trainer.train(epoch)
    trainer.save(epoch, args.output_path)
    if test_data_loader is not None:
        trainer.test(epoch)


