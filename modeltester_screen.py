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

parser.add_argument("-m", "--model", required=True, type=str, help="path to pretrained model to test")
parser.add_argument("-v", "--net_version", type=int, default=0, help="0 for regular, 1 to embed location in UIs, 2 to use layout embedding, 3 to use both, 4 with both but no description, 5 to use both but not train description")
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
if args.net_version in [0,1,2,3]:
    desc_size = 768
else:
    desc_size = 0


orig_model = Screen2Vec(bert_size, additional_ui_size=adus, additional_size_screen=adss, desc_size=desc_size)
predictor = TracePredictor(orig_model, args.net_version)
predictor.load_state_dict(torch.load(args.model))

correct = 0
topone = 0
topfive = 0
topten = 0
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
with open(args.train_data + 'screen_names.json') as f:
    tr_screen_names = json.load(f, encoding='utf-8')


with open(args.test_data + "uis.json") as f:
    te_uis = json.load(f, encoding='utf-8')
with open(args.test_data + "ui_emb.json") as f:
    te_ui_emb = json.load(f, encoding='utf-8')
with open(args.test_data + "descr.json") as f:
    te_descr = json.load(f, encoding='utf-8')
te_descr_emb = np.load(args.test_data + "dsc_emb.npy")
with open(args.test_data + 'screen_names.json') as f:
    te_screen_names = json.load(f, encoding='utf-8')


ui_emb = tr_ui_emb + te_ui_emb
descr_emb = np.concatenate((tr_descr_emb, te_descr_emb))
uis = tr_uis + te_uis
descr = tr_descr + te_descr
screen_names = tr_screen_names + te_screen_names
# ui_emb = tr_ui_emb
# descr_emb = tr_descr_emb
# uis = tr_uis
# descr = tr_descr

if args.net_version not in [0,1]:
    with open(args.train_data + "layout_embeddings.json") as f:
        train_layouts = json.load(f, encoding='utf-8')
    with open(args.test_data + "layout_embeddings.json") as f:
        test_layouts = json.load(f, encoding='utf-8')
    layouts = train_layouts + test_layouts
else:
    layouts = None


dataset = RicoDataset(args.num_predictors, uis, ui_emb, descr, descr_emb, layouts, args.net_version, True, screen_names)       

data_loader = DataLoader(dataset, collate_fn=pad_collate, batch_size=1)
vocab = ScreenVocab(dataset)

end_index = 0
if args.net_version != 5:
    comp = torch.empty(0,bert_size)
else:
    comp = torch.empty(0,bert_size *2)
while end_index != -1:
    vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts , vocab_indx_map, vocab_rvs_indx, end_index = vocab.get_all_screens(end_index, 1024)
    comp_part = predictor.model(vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts, False).squeeze(0)
    comp = torch.cat((comp, comp_part), dim = 0)

comp = comp.detach().numpy()

comp_dict = {}


for emb_idx in range(len(comp)):
    names = vocab.get_name(emb_idx)
    comp_dict[names] = comp[emb_idx].tolist()

mistakes = []

with open('model' + str(args.net_version) + '.json', 'w', encoding='utf-8') as f:
    json.dump(comp_dict, f, indent=4)
i = 0
eek = 0
for data in data_loader:
# run it through the network
    UIs, descr, trace_screen_lengths, index, layouts = data
    #print(i)
    i+=1
    # forward the training stuff (prediction)
    c,result,_ = predictor(UIs, descr, trace_screen_lengths, layouts, False)
    
    # find which vocab vector has the smallest cosine distance
    distances = scipy.spatial.distance.cdist(c.detach().numpy(), comp, "cosine")[0]

    temp = np.argpartition(distances, (0,int(0.01 * len(distances)), int(0.05 * len(distances)), int(0.1 * len(distances))))
    closest_idx = temp[0]
    closest_oneperc = temp[:int(0.01 * len(distances))]
    closest_fiveperc = temp[:int(0.05 * len(distances))]
    closest_tenperc = temp[:int(0.1 * len(distances))]

    if vocab_rvs_indx[index[0][0]][index[0][1]]==closest_idx:
        correct +=1
        topone +=1
        topfive +=1
        topten +=1
    elif vocab_rvs_indx[index[0][0]][index[0][1]] in closest_oneperc:
        topone +=1
        topfive +=1
        topten +=1
    elif vocab_rvs_indx[index[0][0]][index[0][1]] in closest_fiveperc:
        topfive +=1
        topten +=1
    elif vocab_rvs_indx[index[0][0]][index[0][1]] in closest_tenperc:
        topten +=1
    if abs(vocab_rvs_indx[index[0][0]][index[0][1]]-closest_idx) <10 and abs(vocab_rvs_indx[index[0][0]][index[0][1]]-closest_idx) != 0:
        eek+=1
    if vocab_rvs_indx[index[0][0]][index[0][1]] not in closest_fiveperc:
        names = vocab.get_name(vocab_rvs_indx[index[0][0]][index[0][1]])
        bad_names = vocab.get_name(closest_idx)
        mistakes.append((names, bad_names))


    total+=1

with open('mistakes_' + str(args.net_version) + '.json', 'w', encoding='utf-8') as f:
    json.dump(mistakes, f, indent=4)

print(correct/total)
print(topone/total)
print(topfive/total)
print(topten/total)
print(eek)
print(eek/total)


if args.net_version == 4:
    end_index = 0
    comp = torch.empty(0,bert_size*2)
    while end_index != -1:
        vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts , vocab_indx_map, vocab_rvs_indx, end_index = vocab.get_all_screens(end_index, 1024)
        comp_part = predictor.model(vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts).squeeze(0)
        embeddings = torch.cat((comp_part, vocab_descr.squeeze(0)), dim=1)
        comp = torch.cat((comp, embeddings), dim = 0)

    comp = comp.detach().numpy()

    comp_dict = {}


    for emb_idx in range(len(comp)):
        names = vocab.get_name(emb_idx)
        comp_dict[names] = comp[emb_idx].tolist()

    with open('model' + str(args.net_version) + 'descr.json', 'w', encoding='utf-8') as f:
        json.dump(comp_dict, f, indent=4)

from sklearn.cluster import KMeans

num_clusters = 50
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(comp)
assignment = clustering_model.labels_

with open("cluster_output.txt", "w", encoding="utf-8") as f:
    for cl_no in range(num_clusters):
        clustered_words = [str(vocab.get_name(idx)) + "\n" for idx in range(len(assignment)) if assignment[idx] == cl_no ]
        f.write("______________" + "\n")
        f.write(str(cl_no) + ":\n")
        f.write("______________" + "\n")
        f.writelines(clustered_words)