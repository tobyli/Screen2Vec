import argparse
import math
import torch
import torch.nn as nn
import json
import scipy
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Screen2Vec import Screen2Vec
from dataset.dataset import TesterRicoDataset, RicoTrace, RicoScreen
from prediction import BaselinePredictor
from vocab import ScreenVocab

class BaselineDataset(Dataset):
    '''
    has traces, which have screens
    '''
    def __init__(self, embeddings, n):
        self.n = n
        self.traces = self.load_traces(embeddings)
        self.trace_loc_index = []
        self.embeddings = []
        self.load_indices()

    def __getitem__(self, index):
        indexed_trace = self.traces[index]
        # not added unless there are at least n screens in the trace
        traces = []
        if len(indexed_trace) >= self.n:
            starting_index = random.randint(0, len(indexed_trace)-self.n)
            screens = indexed_trace[starting_index:starting_index+self.n-2]
            target_index = self.get_overall_index(index, starting_index+self.n -1)
        return torch.tensor(screens), torch.tensor(target_index)
    
    def __len__(self):
        return len(self.traces)

    def load_traces(self,emb):
        traces = []
        for trace in emb:
            if len(trace) >= self.n:
                traces.append(trace)
        return traces

    def load_indices(self):
        overall_index = 0
        for trace in self.traces:
            self.embeddings += trace
            self.trace_loc_index.append([overall_index + i for i in range(len(trace))])
            overall_index += len(trace)

    def get_overall_index(self, trace_idx, screen_idx):
        return self.trace_loc_index[trace_idx][screen_idx]
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", required=True, type=str, help="path to pretrained model to test")
parser.add_argument("-v", "--net_version", type=int, default=0, help="0 for regular, 1 to embed location in UIs, 2 to use layout embedding, 3 to use both, 4 with both but no description, 5 to use both but not train description")
parser.add_argument("-d", "--data", required=True, type=str, default=None, help="prefix of precomputed data")
parser.add_argument("-n", "--num_predictors", type=int, default=10, help="number of other labels used to predict one")
args = parser.parse_args()


bert_size = 768
if args.net_version in [0,2,6,8]:
    adus = 0
else:
    # case where coordinates are part of UI rnn
    adus = 4
if args.net_version in [0,1,6,7]:
    adss = 0
else:
    # case where screen layout vec is used
    adss = 64


orig_model = Screen2Vec(bert_size, additional_ui_size=adus, additional_size_screen=adss, net_version=args.net_version)
predictor = TracePredictor(orig_model, args.net_version)
predictor.load_state_dict(torch.load(args.model))

correct = 0
toppointzeroone = 0
toppointone = 0
topone = 0
topfive = 0
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

if args.net_version not in [0,1,6,7]:
    with open(args.data + "layout_embeddings.json") as f:
        layouts = json.load(f, encoding='utf-8')
else:
    layouts = None


dataset = TesterRicoDataset(args.num_predictors, uis, ui_emb, descr, descr_emb, layouts, args.net_version, True, screen_names)       

data_loader = DataLoader(dataset, collate_fn=pad_collate, batch_size=1)
vocab = ScreenVocab(dataset)

# end_index = 0
# if args.net_version not in [5]:
#     comp = torch.empty(0,bert_size).detach()
# else:
#     comp = torch.empty(0,bert_size *2).detach()

# all_layouts = torch.empty(0,64)
# all_avg_embeddings = torch.empty(0,768 + adus)

# while end_index != -1:
#     vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts, vocab_indx_map, vocab_rvs_indx, end_index = vocab.get_all_screens(end_index, 1024)
#     comp_part = predictor.model(vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts, False).squeeze(0)
#     comp = torch.cat((comp, comp_part.detach()), dim = 0)

# comp = comp.detach().numpy()


# i = 0
# eek = 0
# for data in data_loader:
# # run it through the network
#     UIs, descr, trace_screen_lengths, index, layouts = data
#     #print(i)
#     i+=1
#     # forward the training stuff (prediction)
#     c,result,_ = predictor(UIs, descr, trace_screen_lengths, layouts, False)
    
#     # find which vocab vector has the smallest cosine distance
#     distances = scipy.spatial.distance.cdist(c.detach().numpy(), comp, "cosine")[0]

#     temp = np.argpartition(distances, (0,int(0.01 * len(distances)), int(0.05 * len(distances)), int(0.1 * len(distances))))
#     closest_idx = temp[0]
#     closest_oneperc = temp[:int(0.01 * len(distances))]
#     closest_fiveperc = temp[:int(0.05 * len(distances))]
#     closest_tenperc = temp[:int(0.1 * len(distances))]

#     if vocab_rvs_indx[index[0][0]][index[0][1]]==closest_idx:
#         correct +=1
#         topone +=1
#         topfive +=1
#         topten +=1
#     elif vocab_rvs_indx[index[0][0]][index[0][1]] in closest_oneperc:
#         topone +=1
#         topfive +=1
#         topten +=1
#     elif vocab_rvs_indx[index[0][0]][index[0][1]] in closest_fiveperc:
#         topfive +=1
#         topten +=1
#     elif vocab_rvs_indx[index[0][0]][index[0][1]] in closest_tenperc:
#         topten +=1
#     if abs(vocab_rvs_indx[index[0][0]][index[0][1]]-closest_idx) <10 and abs(vocab_rvs_indx[index[0][0]][index[0][1]]-closest_idx) != 0:
#         eek+=1
#     if vocab_rvs_indx[index[0][0]][index[0][1]] not in closest_fiveperc:
#         names = vocab.get_name(vocab_rvs_indx[index[0][0]][index[0][1]])
#         bad_names = vocab.get_name(closest_idx)


#     total+=1



# print(str(correct/total) + " of the predictions were exactly correct")
# print(str(topone/total) + " of the predictions were in the top 1%")
# print(str(topfive/total) + " of the predictions were in the top 5%")
# print(str(topten/total) + " of the predictions were in the top 10%")
# print(str(eek/total) + " of the predictions were correct, but predicted a screen nearby in the trace")


if args.net_version in [4,6,7,8]:
    end_index = 0
    comp = torch.empty(0,bert_size*2)
    while end_index != -1:
        vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts , vocab_indx_map, vocab_rvs_indx, end_index = vocab.get_all_screens(end_index, 1024)
        comp_part = predictor.model(vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts).squeeze(0)
        embeddings = torch.cat((comp_part, vocab_descr.squeeze(0)), dim=1)
        comp = torch.cat((comp, embeddings), dim = 0)

    comp = comp.detach()


    error = nn.MSELoss()

    i = 0
    eek = 0
    total_se = 0
    total_vector_lengths = 0
    for data in data_loader:
        UIs, descr, trace_screen_lengths, index, layouts = data
    # run it through the network
        UIs, descr, trace_screen_lengths, index, layouts = data
        i+=len(index)
        # forward the training stuff (prediction)
        c,result,_ = predictor(UIs, descr, trace_screen_lengths, layouts, False)
        descr = torch.narrow(descr,1,0,1).squeeze(1)
        c = torch.cat((c,descr),dim=-1)
        # find which vocab vector has the smallest cosine distance
        for idx in range(len(index)):
            correct_index = vocab_rvs_indx[index[idx][0]][index[idx][1]]
            #print(c.size())
            #print(comp[correct_index].size())
            diff = c.detach()[idx]-comp[correct_index]
            sqer = sum(diff**2)
            total_se += sqer
            total_vector_lengths += np.linalg.norm(diff)
            distances = scipy.spatial.distance.cdist(c.detach()[idx].unsqueeze(dim=0), comp, "cosine")[0]
        
            
            temp = np.argpartition(distances, (0,int(0.01 * len(distances)), int(0.05 * len(distances)), int(0.1 * len(distances))))
            closest_idx = temp[0]
            closest_pointzerooneperc = temp[:int(0.0001 * len(distances))]
            closest_pointoneperc = temp[:int(0.001 * len(distances))]
            closest_oneperc = temp[:int(0.01 * len(distances))]
            closest_fiveperc = temp[:int(0.05 * len(distances))]

            if correct_index==closest_idx:
                correct +=1
                toppointzeroone+=1
                toppointone +=1
                topone +=1
                topfive +=1
            elif correct_index in closest_pointzerooneperc:
                toppointzeroone +=1
                toppointone +=1
                topone +=1
                topfive +=1
            elif correct_index in closest_pointoneperc:
                toppointone +=1
                topone +=1
                topfive +=1
            elif correct_index in closest_oneperc:
                topone +=1
                topfive +=1
            elif correct_index in closest_fiveperc:
                topfive +=1

            total+=1

    total_rmse = math.sqrt(total_se/i)/(total_vector_lengths/i)
    print(str(correct/total) + " of the predictions were exactly correct")
    print(str(toppointzeroone/total) + " of the predictions were in the top 0.01%")
    print(str(toppointone/total) + " of the predictions were in the top 0.1%")
    print(str(topone/total) + " of the predictions were in the top 1%")
    print(str(topfive/total) + " of the predictions were in the top 5%")
    print("rmse error is: " + str(total_rmse/i))

