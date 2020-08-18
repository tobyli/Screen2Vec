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
from dataset.dataset import TesterRicoDataset, RicoTrace, RicoScreen
from sentence_transformers import SentenceTransformer
from prediction import TracePredictor
from vocab import ScreenVocab



def pad_collate(batch):
    UIs = [seq[0] for trace in batch for seq in trace]
    descr = torch.tensor([seq[1] for trace in batch for seq in trace])
    correct_indices = [seq[2] for trace in batch for seq in trace]
    if batch[0][0][3]:
        layouts = torch.FloatTensor([seq[3] for trace in batch for seq in trace])
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
parser.add_argument("-d", "--data", required=True, type=str, default=None, help="prefix of precomputed data")
parser.add_argument("-n", "--num_predictors", type=int, default=10, help="number of other labels used to predict one")
args = parser.parse_args()


bert_size = 768
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


orig_model = Screen2Vec(bert_size, additional_ui_size=adus, additional_size_screen=adss, net_version=args.net_version)
predictor = TracePredictor(orig_model, args.net_version)
predictor.load_state_dict(torch.load(args.model))

correct = 0
topone = 0
topfive = 0
topten = 0
total = 0

with open(args.data + "uis.json") as f:
    uis = json.load(f, encoding='utf-8')

ui_emb = []
try:
    for i in range(10):
        with open(args.data + str(i) + "_ui_emb.json") as f:
            ui_emb += json.load(f, encoding='utf-8')
        print(i)
except FileNotFoundError as e:
    with open(args.data + "ui_emb.json") as f:
            ui_emb += json.load(f, encoding='utf-8')

with open(args.data + "descr.json") as f:
    descr = json.load(f, encoding='utf-8')
descr_emb = np.load(args.data + "dsc_emb.npy")

with open(args.data + 'screen_names.json') as f:
    screen_names = json.load(f, encoding='utf-8')

if args.net_version not in [0,1,6]:
    with open(args.data + "layout_embeddings.json") as f:
        layouts = json.load(f, encoding='utf-8')
else:
    layouts = None


dataset = TesterRicoDataset(args.num_predictors, uis, ui_emb, descr, descr_emb, layouts, args.net_version, True, screen_names)       

data_loader = DataLoader(dataset, collate_fn=pad_collate, batch_size=1)
vocab = ScreenVocab(dataset)

end_index = 0
if args.net_version not in [5]:
    comp = torch.empty(0,bert_size)
else:
    comp = torch.empty(0,bert_size *2)

all_layouts = torch.empty(0,64)
all_avg_embeddings = torch.empty(0,768 + adus)

while end_index != -1:
    vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts, vocab_indx_map, vocab_rvs_indx, end_index = vocab.get_all_screens(end_index, 1024)
    comp_part = predictor.model(vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts, False).squeeze(0)
    comp = torch.cat((comp, comp_part), dim = 0)
    # all_layouts = torch.cat((all_layouts, vocab_layouts.squeeze(0)), dim=0)
    # sum_embeddings = vocab_UIs.squeeze(0).sum(dim=0)
    # vocab_trace_screen_lengths = vocab_trace_screen_lengths.squeeze(0)
    # avg_embeddings = torch.stack([sum_embeddings[x]/int(vocab_trace_screen_lengths[x]) for x in range(len(sum_embeddings))])
    # all_avg_embeddings = torch.cat((all_avg_embeddings,avg_embeddings), dim=0)

comp = comp.detach().numpy()
# all_layouts = all_layouts.detach().numpy()
# all_avg_embeddings = avg_embeddings.detach().numpy()

comp_dict = {}
# layout_dict = {}
# avg_dict = {}

for emb_idx in range(len(comp)):
    names = vocab.get_name(emb_idx)
    comp_dict[names] = comp[emb_idx].tolist()
    # layout_dict[names] = all_layouts[emb_idx].tolist()
    # avg_dict[names] = all_avg_embeddings[emb_idx].tolist()

mistakes = []

with open('model' + str(args.net_version) + 'full.json', 'w', encoding='utf-8') as f:
    json.dump(comp_dict, f, indent=4)
# with open('model' + str(args.net_version) + 'avgui.json', 'w', encoding='utf-8') as f:
#     json.dump(avg_dict, f, indent=4)
# with open('model' + str(args.net_version) + 'layout.json', 'w', encoding='utf-8') as f:
#     json.dump(layout_dict, f, indent=4)


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

print(str(correct/total) + " of the predictions were exactly correct")
print(str(topone/total) + " of the predictions were in the top 1%")
print(str(topfive/total) + " of the predictions were in the top 5%")
print(str(topten/total) + " of the predictions were in the top 10%")
print(str(eek/total) + " of the predictions were correct, but predicted a screen nearby in the trace")


if args.net_version in [4,6]:
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


    i = 0
    eek = 0
    for data in data_loader:
    # run it through the network
        UIs, descr, trace_screen_lengths, index, layouts = data
        #print(i)
        i+=1
        # forward the training stuff (prediction)
        c,result,_ = predictor(UIs, descr, trace_screen_lengths, layouts, False)
        descr = torch.narrow(descr,1,0,1).squeeze(1)
        c = torch.cat((c,descr),dim=-1)
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

    print(str(correct/total) + " of the predictions were exactly correct")
    print(str(topone/total) + " of the predictions were in the top 1%")
    print(str(topfive/total) + " of the predictions were in the top 5%")
    print(str(topten/total) + " of the predictions were in the top 10%")
    print(str(eek/total) + " of the predictions were correct, but predicted a screen nearby in the trace")


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