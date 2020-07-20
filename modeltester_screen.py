import argparse
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
import scipy
import numpy as np
from torch.utils.data import DataLoader
from Screen2Vec import Screen2Vec
from pretrainer import Screen2VecTrainer
from dataset.dataset import RicoDataset, RicoTrace, RicoScreen
from sentence_transformers import SentenceTransformer
from prediction import TracePredictor
from vocab import ScreenVocab



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
    return UIs, descr, torch.tensor(trace_screen_lengths), correct_indices

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", required=True, type=str, help="path to pretrained model to test")
parser.add_argument("-r", "--range", type=float, default=0.1, help="what proportion of results to look in")
parser.add_argument("-v", "--net-version", type=int, default=0, help="0 for regular, 1 to embed location in UIs, 2 to use layout embedding, and 3 to use both")
parser.add_argument("-c", "--train_data", required=True, type=str, default=None, help="prefix of precomputed data to train model")
parser.add_argument("-t", "--test_data", required=False, type=str, default=None, help="prefix of precomputed data to test model")
parser.add_argument("-f", "--folder", required=True, type=str, help="path to Screen2Vec folder")
parser.add_argument("-n", "--num_predictors", type=int, default=10, help="number of other labels used to predict one")
args = parser.parse_args()


bert_size = 768
if args.net_version in [0,2]:
    adus = 0
else:
    # case where coordinates are part of UI rnn
    adus = 4
if args.net_version in [0,1]:
    adss = 0
else:
    # case where screen layout vec is used
    adss = 64

orig_model = Screen2Vec(bert_size, num_classes=24, additional_ui_size=adus, additional_size_screen=adss)
predictor = TracePredictor(orig_model)
predictor.load_state_dict(torch.load(args.model))

correct = 0
total = 0

with open(args.train_data + "uis.json") as f:
    tr_uis = json.load(f, encoding='utf-8')
tr_ui_emb = []
for i in range(10):
    print(i)
    with open(args.train_data + str(i) + "_ui_emb.json") as f:
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


ui_emb = tr_ui_emb + te_ui_emb
descr_emb = np.concatenate((tr_descr_emb, te_descr_emb))
uis = tr_uis + te_uis
descr = tr_descr + te_descr
# ui_emb = tr_ui_emb
# descr_emb = tr_descr_emb
# uis = tr_uis
# descr = tr_descr

if args.net_version in [2,3]:
    with open(args.test_data + "layout_emb_idx.json") as f:
        layout_emb_idx = json.load(f, encoding='utf-8')
    layouts = np.load(args.folder + "/Screen2Vec/ui_layout_vectors/ui_vectors.npy")
else:
    layout_emb_idx = None
    layouts = None


dataset = RicoDataset(args.num_predictors, uis, ui_emb, descr, descr_emb, layout_emb_idx, layouts, args.net_version)       

data_loader = DataLoader(dataset, collate_fn=pad_collate, batch_size=1)
vocab = ScreenVocab(dataset)

vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_indx_map, vocab_rvs_indx = vocab.get_all_screens()
comp = predictor.model(vocab_UIs, vocab_descr, vocab_trace_screen_lengths).squeeze(0)

i = 0
for data in data_loader:
# run it through the network
    UIs, descr, trace_screen_lengths, index = data
    print(i)
    i+=1
    # forward the training stuff (prediction)
    c,result = predictor(UIs, descr, trace_screen_lengths, False)
    
    # find which vocab vector has the smallest cosine distance
    distances = scipy.spatial.distance.cdist(c.detach().numpy(), comp.detach().numpy(), "cosine")[0]

    temp = np.argpartition(distances, int(args.range * len(distances)))
    closest_idx = temp[:int(args.range * len(distances))]

    if int(vocab_rvs_indx[index[0][0]][index[0][1]]) in closest_idx:
        correct +=1
    total+=1


print(correct/total)


    # from sklearn.cluster import KMeans

    # corpus = [screen.labeled_text for screen in dataset.screens]
    # corpus_text = [bundle[0] for text in corpus for bundle in text] 
    # corpus_class = torch.tensor([bundle[1] for text in corpus for bundle in text])
    # corpus_embeddings = model.model((corpus_text, corpus_class)).detach().numpy()

    # num_clusters = 50
    # clustering_model = KMeans(n_clusters=num_clusters)
    # clustering_model.fit(corpus_embeddings)
    # assignment = clustering_model.labels_

    # with open("cluster_output.txt", "w", encoding="utf-8") as f:
    #     for cl_no in range(num_clusters):
    #         clustered_words = [corpus_text[idx] + "\n" for idx in range(len(assignment)) if assignment[idx] == cl_no ]
    #         print(cl_no)
    #         print(clustered_words[:10])
    #         f.write(str(cl_no) + ":\n")
    #         f.writelines(clustered_words)