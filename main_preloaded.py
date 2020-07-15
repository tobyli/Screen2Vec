import argparse
import json
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from Screen2Vec import Screen2Vec
from pretrainer import Screen2VecTrainer
from dataset.dataset import RicoDataset, RicoTrace, RicoScreen
from sentence_transformers import SentenceTransformer
from prediction import TracePredictor
from vocab import ScreenVocab
from UI_embedding.plotter import plot_loss


def pad_collate(batch):
    UIs = [trace[0] for trace in batch]
    descr = torch.tensor([trace[1] for trace in batch])
    correct_indices = [trace[2] for trace in batch]

    trace_screen_lengths = []
    for trace_idx in range(len(UIs)):
        #UIs[trace_idx] has dimensions len(trace) x len(screen) x bert emb length
        screen_lengths = [len(screen) for screen in UIs[trace_idx]]
        trace_screen_lengths.append(screen_lengths)
        UIs[trace_idx] = torch.nn.utils.rnn.pad_sequence(UIs[trace_idx])
    UIs = torch.nn.utils.rnn.pad_sequence(UIs)
    UIs = UIs.transpose(0,1) #may want to not do this?
    print(UIs.size())
    return UIs, descr, torch.tensor(trace_screen_lengths), correct_indices

parser = argparse.ArgumentParser()


parser.add_argument("-c", "--train_data", required=True, type=str, default=None, help="prefix of precomputed data to train model")
parser.add_argument("-t", "--test_data", required=False, type=str, default=None, help="prefix of precomputed data to test model")
parser.add_argument("-o", "--output_path", required=True, type=str, help="where to store model")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="traces in a batch")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-n", "--num_predictors", type=int, default=10, help="number of other labels used to predict one")
parser.add_argument("-l", "--loss", type=int, default=1, help="1 to use cosine embedding loss, 0 to use softmax dot product")
parser.add_argument("-r", "--rate", type=float, default=0.001, help="learning rate")
parser.add_argument("-s", "--neg_samp", type=int, default=128, help="number of negative samples")

args = parser.parse_args()

bert = SentenceTransformer('bert-base-nli-mean-tokens')
bert_size = 768


# with open(args.train_data + "uis.json") as f:
#     tr_uis = json.load(f, encoding='utf-8')

tr_ui_emb = []
for i in range(10):
    print(i)
    with open(args.train_data + str(i) + "_ui_emb.json") as f:
        tr_ui_emb += json.load(f, encoding='utf-8')

# with open(args.train_data + "descr.json") as f:
#     tr_descr = json.load(f, encoding='utf-8')
tr_descr_emb = np.load(args.train_data + "dsc_emb.npy")

# with open(args.test_data + "uis.json") as f:
#     te_uis = json.load(f, encoding='utf-8')
with open(args.test_data + "ui_emb.json") as f:
    te_ui_emb = json.load(f, encoding='utf-8')

# with open(args.test_data + "descr.json") as f:
#     te_descr = json.load(f, encoding='utf-8')
te_descr_emb = np.load(args.test_data + "dsc_emb.npy")


train_dataset = RicoDataset(args.num_predictors, tr_ui_emb, tr_descr_emb)
test_dataset = RicoDataset(args.num_predictors, te_ui_emb, te_descr_emb)

vocab = ScreenVocab(train_dataset)

train_data_loader = DataLoader(train_dataset, collate_fn=pad_collate, batch_size=args.batch_size)
test_data_loader = DataLoader(test_dataset, collate_fn=pad_collate, batch_size=args.batch_size)

# if args.test_dataset:
#     test_dataset = RicoDataset(loaded_model.model, args.test_dataset, args.num_predictors)
#     test_data_loader = DataLoader(test_dataset, collate_fn=pad_collate, batch_size=args.batch_size)
# else:
#     test_data_loader = None


#test_data_loader = None

model = Screen2Vec(bert_size)
predictor = TracePredictor(model)

trainer = Screen2VecTrainer(predictor, vocab, train_data_loader, test_data_loader, args.rate, args.neg_samp)

predictor.cuda()
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
    plot_loss(train_loss_data, test_loss_data)

