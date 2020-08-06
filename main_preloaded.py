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
    """
    collate function that handles variable numbers of UIs in a screen
    """
    UIs = [trace[0] for trace in batch]
    descr = torch.FloatTensor([trace[1] for trace in batch])
    correct_indices = [trace[2] for trace in batch]
    if batch[0][3]:
        layouts = torch.FloatTensor([trace[3] for trace in batch])
    else:
        layouts = None

    trace_screen_lengths = []
    for trace_idx in range(len(UIs)):
        #UIs[trace_idx] has dimensions len(trace) x len(screen) x bert emb length
        screen_lengths = [len(screen) for screen in UIs[trace_idx]]
        trace_screen_lengths.append(screen_lengths)
        UIs[trace_idx] = torch.nn.utils.rnn.pad_sequence(UIs[trace_idx])
    UIs = torch.nn.utils.rnn.pad_sequence(UIs)
    UIs = UIs.transpose(0,1) #may want to not do this?
    return UIs, descr, torch.tensor(trace_screen_lengths), correct_indices, layouts

parser = argparse.ArgumentParser()


parser.add_argument("-c", "--train_data", required=True, type=str, default=None, help="prefix of precomputed data to train model")
parser.add_argument("-t", "--test_data", required=False, type=str, default=None, help="prefix of precomputed data to test model")
parser.add_argument("-o", "--output_path", required=True, type=str, help="where to store model")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="traces in a batch")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-n", "--num_predictors", type=int, default=10, help="number of other labels used to predict one")
parser.add_argument("-r", "--rate", type=float, default=0.001, help="learning rate")
parser.add_argument("-s", "--neg_samp", type=int, default=128, help="number of negative samples")
parser.add_argument("-a", "--prev_model", type=str, default=None, help="previously trained model to start training from")
# parser.add_argument("-f", "--folder", type=str, default="", help="path to Screen2Vec folder")
parser.add_argument("-v", "--net_version", type=int, default=0, help="0 for regular, 1 to embed location in UIs, 2 to use layout embedding, 3 to use both, 4 with both but no description, 5 to use both but not train description")


args = parser.parse_args()

bert_size = 768

# preloading starts here
with open(args.train_data + "uis.json") as f:
    tr_uis = json.load(f, encoding='utf-8')

tr_ui_emb = []
try:
    for i in range(10):
        with open(args.train_data + str(i) + "_ui_emb.json") as f:
            tr_ui_emb += json.load(f, encoding='utf-8')
        print(i)
except FileNotFoundError as e:
    with open(args.train_data + "ui_emb.json") as f:
            tr_ui_emb += json.load(f, encoding='utf-8')

with open(args.train_data + "descr.json") as f:
    tr_descr = json.load(f, encoding='utf-8')
tr_descr_emb = np.load(args.train_data + "dsc_emb.npy")

with open(args.test_data + "uis.json") as f:
    te_uis = json.load(f, encoding='utf-8')
with open(args.test_data + "ui_emb.json") as f:
    te_ui_emb = json.load(f, encoding='utf-8')

with open(args.test_data + "descr.json") as f:
    te_descr = json.load(f, encoding='utf-8')
te_descr_emb = np.load(args.test_data + "dsc_emb.npy")

if args.net_version not in [0,1,6]:
    with open(args.train_data + "layout_embeddings.json") as f:
        train_layouts = json.load(f, encoding='utf-8')
    with open(args.test_data + "layout_embeddings.json") as f:
        test_layouts = json.load(f, encoding='utf-8')
else:
    train_layouts = None
    test_layouts = None
# preloading ends here

train_dataset = RicoDataset(args.num_predictors, tr_uis, tr_ui_emb, tr_descr, tr_descr_emb, train_layouts, args.net_version)
test_dataset = RicoDataset(args.num_predictors, te_uis, te_ui_emb, te_descr, te_descr_emb, test_layouts, args.net_version)

vocab_train = ScreenVocab(train_dataset)
vocab_test = ScreenVocab(test_dataset)

train_data_loader = DataLoader(train_dataset, collate_fn=pad_collate, batch_size=args.batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, collate_fn=pad_collate, batch_size=args.batch_size, shuffle=True)


#handle different versions of network here
if args.net_version in [0,2,6]:
    adus = 0
else:
    # case where coordinates are part of UI rnn
    adus = 4
if args.net_version in [0,1,6]:
    adss = 0
else:
    # case where screen layout vec is used
    adss = 64
if args.net_version in [0,1,2,3]:
    desc_size = 768
else:
    # no description in training case
    desc_size = 0

# generate models
model = Screen2Vec(bert_size, additional_ui_size=adus, additional_size_screen=adss, desc_size=desc_size)
predictor = TracePredictor(model, args.net_version)
predictor.cuda()
if args.prev_model:
    predictor.load_state_dict(torch.load(args.prev_model))
trainer = Screen2VecTrainer(predictor, vocab_train, vocab_test, train_data_loader, test_data_loader, args.rate, args.neg_samp)


# training occurs below
test_loss_data = []
train_loss_data = []
for epoch in tqdm.tqdm(range(args.epochs)):
    print("--------")
    print(str(epoch) + " train loss:")
    train_loss = trainer.train(epoch)
    print(train_loss)
    print("--------")
    train_loss_data.append(train_loss)
    if test_data_loader is not None:
        print("--------")
        print(str(epoch) + " test loss:")
        test_loss = trainer.test(epoch)
        print(test_loss)
        print("--------")
        test_loss_data.append(test_loss)
    if (epoch%20)==0:
        print("saved on epoch " + str(epoch))
        trainer.save(epoch, args.output_path)
plot_loss(train_loss_data, test_loss_data, args.output_path)
trainer.save(args.epochs, args.output_path)

