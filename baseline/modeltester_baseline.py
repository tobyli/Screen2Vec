import argparse
import math
import torch
import tqdm
import torch.nn as nn
import json
import scipy
import numpy as np
from torch.utils.data import Dataset, DataLoader
from prediction import BaselinePredictor
from sentence_transformers import SentenceTransformer

# modeltester for a baseline model

class TesterBaselineDataset(Dataset):
    '''
    has traces, which have screens
    '''
    def __init__(self, embeddings, n):
        self.n = n
        self.traces = self.load_traces(embeddings)
        self.trace_loc_index = []
        self.embeddings = self.load_indices()

    def __getitem__(self, index):
        indexed_trace = self.traces[index]
        # not added unless there are at least n screens in the trace
        traces = []
        indices = []
        if len(indexed_trace) >= self.n:
            starting_indices = range(0, len(indexed_trace)-self.n+1)
            for start_idx in starting_indices:
                screens = indexed_trace[start_idx:start_idx+self.n-1]
                target_index = self.get_overall_index(index, start_idx+self.n -1)
                traces.append(screens)
                indices.append(target_index)
        return torch.tensor(traces), torch.tensor(indices)
    
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
        embeddings = []
        for trace in self.traces:
            embeddings += trace
            self.trace_loc_index.append([overall_index + i for i in range(len(trace))])
            overall_index += len(trace)
        return np.asarray(embeddings)

    def get_overall_index(self, trace_idx, screen_idx):
        return self.trace_loc_index[trace_idx][screen_idx]

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--precomp", required=True, type=str, help="path to precomputed embeddings")
parser.add_argument("-n", "--num_predictors", type=int, default=4, help="number of other labels used to predict one")
parser.add_argument("-l", "--layout_predictor", required=True, type=str, help="path to layout prediction model")
parser.add_argument("-v", "--visual_predictor", required=True, type=str, help="path to visual prediction model")
parser.add_argument("-t", "--text_predictor", required=True, type=str, help="path to text prediction model")
args = parser.parse_args()

with open(args.precomp + "layout_eval.json") as f:
    layout_emb = json.load(f, encoding='utf-8')

predictor = BaselinePredictor(len(layout_emb[0][0]))
predictor.load_state_dict(torch.load(args.layout_predictor))

correct = 0
toppointzeroone = 0
toppointone = 0
topone = 0
topfive = 0
total = 0

dataset = TesterBaselineDataset(layout_emb, args.num_predictors)       
data_loader = DataLoader(dataset, batch_size=1)

data_itr = tqdm.tqdm(enumerate(data_loader),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

emb_len = len(dataset.embeddings)
i = 0
total_se = 0
total_vector_lengths = 0
for data_idx, data in data_itr:
    trace, index = data
    trace = trace.squeeze(0)
    index = index.squeeze(0)
    result = predictor(trace)
    i+=len(index)
    # forward the training stuff (prediction)
    # find which vocab vector has the smallest cosine distance
    distances = scipy.spatial.distance.cdist(result.detach(), dataset.embeddings, "cosine")
    for idx in range(len(index)):
        target = index[idx]
        diff = result[idx].detach()-torch.tensor(dataset.embeddings[target])
        sqer = sum(diff**2)
        total_se += sqer
        total_vector_lengths += np.linalg.norm(torch.tensor(dataset.embeddings[target]).detach())

        temp = np.argpartition(distances[idx], (0, int(0.0001 * emb_len), int(0.001 * emb_len), int(0.01 * emb_len), int(0.05 * emb_len)))
        closest_idx = temp[0]
        closest_pointzerooneperc = temp[:int(0.0001 * emb_len)]
        closest_pointoneperc = temp[:int(0.001 * emb_len)]
        closest_oneperc = temp[:int(0.01 * emb_len)]
        closest_fiveperc = temp[:int(0.05 * emb_len)]

        if int(target)==closest_idx:
            correct +=1
            toppointzeroone+=1
            toppointone +=1
            topone +=1
            topfive +=1
        elif int(target) in closest_pointzerooneperc:
            toppointzeroone +=1
            toppointone +=1
            topone +=1
            topfive +=1
        elif int(target) in closest_pointoneperc:
            toppointone +=1
            topone +=1
            topfive +=1
        elif int(target) in closest_oneperc:
            topone +=1
            topfive +=1
        elif int(target) in closest_fiveperc:
            topfive +=1

        total+=1

print("For text baseline:")
total_rmse = math.sqrt(total_se/i)/(total_vector_lengths/i)
print(str(correct/total) + " of the predictions were exactly correct")
print(str(toppointzeroone/total) + " of the predictions were in the top 0.01%")
print(str(toppointone/total) + " of the predictions were in the top 0.1%")
print(str(topone/total) + " of the predictions were in the top 1%")
print(str(topfive/total) + " of the predictions were in the top 5%")
print("rmse error is: " + str(total_rmse), flush=True)

bert_size = 768

predictor = BaselinePredictor(bert_size)
predictor.load_state_dict(torch.load(args.text_predictor))

correct = 0
toppointzeroone = 0
toppointone = 0
topone = 0
topfive = 0
total = 0

with open(args.precomp + "text_eval.json") as f:
    text_emb = json.load(f, encoding='utf-8')

dataset = TesterBaselineDataset(text_emb, args.num_predictors)       
data_loader = DataLoader(dataset, batch_size=1)

emb_len = len(dataset.embeddings)

data_itr = tqdm.tqdm(enumerate(data_loader),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

i = 0
total_se = 0
total_vector_lengths = 0
for data_idx, data in data_itr:
    trace, index = data
    trace = trace.squeeze(0)
    index = index.squeeze(0)
    result = predictor(trace)
    i+=len(index)
    # forward the training stuff (prediction)
    # find which vocab vector has the smallest cosine distance
    distances = scipy.spatial.distance.cdist(result.detach(), dataset.embeddings, "cosine")
    for idx in range(len(index)):
        target = index[idx]
        diff = result[idx].detach()-torch.tensor(dataset.embeddings[target])
        sqer = sum(diff**2)
        total_se += sqer
        total_vector_lengths += np.linalg.norm(torch.tensor(dataset.embeddings[target]).detach())

        temp = np.argpartition(distances[idx], (0, int(0.0001 * emb_len), int(0.001 * emb_len), int(0.01 * emb_len), int(0.05 * emb_len)))
        closest_idx = temp[0]
        closest_pointzerooneperc = temp[:int(0.0001 * emb_len)]
        closest_pointoneperc = temp[:int(0.001 * emb_len)]
        closest_oneperc = temp[:int(0.01 * emb_len)]
        closest_fiveperc = temp[:int(0.05 * emb_len)]
        if int(target)==closest_idx:
            correct +=1
            toppointzeroone+=1
            toppointone +=1
            topone +=1
            topfive +=1
        elif int(target) in closest_pointzerooneperc:
            toppointzeroone +=1
            toppointone +=1
            topone +=1
            topfive +=1
        elif int(target) in closest_pointoneperc:
            toppointone +=1
            topone +=1
            topfive +=1
        elif int(target) in closest_oneperc:
            topone +=1
            topfive +=1
        elif int(target) in closest_fiveperc:
            topfive +=1

        total+=1

print("For text baseline:")
total_rmse = math.sqrt(total_se/i)/(total_vector_lengths/i)
print(str(correct/total) + " of the predictions were exactly correct")
print(str(toppointzeroone/total) + " of the predictions were in the top 0.01%")
print(str(toppointone/total) + " of the predictions were in the top 0.1%")
print(str(topone/total) + " of the predictions were in the top 1%")
print(str(topfive/total) + " of the predictions were in the top 5%")
print("rmse error is: " + str(total_rmse), flush=True)




with open(args.precomp + "visual_eval.json") as f:
    vis_emb = json.load(f, encoding='utf-8')

predictor = BaselinePredictor(len(vis_emb[0][0]))
predictor.load_state_dict(torch.load(args.visual_predictor))

correct = 0
toppointzeroone = 0
toppointone = 0
topone = 0
topfive = 0
total = 0

dataset = TesterBaselineDataset(vis_emb, args.num_predictors)       
data_loader = DataLoader(dataset, batch_size=1)

data_itr = tqdm.tqdm(enumerate(data_loader),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

i = 0
total_se = 0
total_vector_lengths = 0
for idx, data in data_itr:
    trace, index = data
    trace = trace.squeeze(0)
    index = index.squeeze(0)
    result = predictor(trace)
    i+=len(index)
    # forward the training stuff (prediction)
    # find which vocab vector has the smallest cosine distance
    trace, index = data
    trace = trace.squeeze(0)
    index = index.squeeze(0)
    result = predictor(trace)
    i+=len(index)
    # forward the training stuff (prediction)
    # find which vocab vector has the smallest cosine distance
    distances = scipy.spatial.distance.cdist(result.detach(), dataset.embeddings, "cosine")
    for idx in range(len(index)):
        target = index[idx]
        diff = result[idx].detach()-torch.tensor(dataset.embeddings[target])
        sqer = sum(diff**2)
        total_se += sqer
        total_vector_lengths += np.linalg.norm(torch.tensor(dataset.embeddings[target]).detach())

        temp = np.argpartition(distances[idx], (0, int(0.0001 * emb_len), int(0.001 * emb_len), int(0.01 * emb_len), int(0.05 * emb_len)))
        closest_idx = temp[0]
        closest_pointzerooneperc = temp[:int(0.0001 * emb_len)]
        closest_pointoneperc = temp[:int(0.001 * emb_len)]
        closest_oneperc = temp[:int(0.01 * emb_len)]
        closest_fiveperc = temp[:int(0.05 * emb_len)]
        if int(target)==closest_idx:
            correct +=1
            toppointzeroone+=1
            toppointone +=1
            topone +=1
            topfive +=1
        elif int(target) in closest_pointzerooneperc:
            toppointzeroone +=1
            toppointone +=1
            topone +=1
            topfive +=1
        elif int(target) in closest_pointoneperc:
            toppointone +=1
            topone +=1
            topfive +=1
        elif int(target) in closest_oneperc:
            topone +=1
            topfive +=1
        elif int(target) in closest_fiveperc:
            topfive +=1

        total+=1

print("For visual baseline:")
total_rmse = math.sqrt(total_se/i)/(total_vector_lengths/i)
print(str(correct/total) + " of the predictions were exactly correct")
print(str(toppointzeroone/total) + " of the predictions were in the top 0.01%")
print(str(toppointone/total) + " of the predictions were in the top 0.1%")
print(str(topone/total) + " of the predictions were in the top 1%")
print(str(topfive/total) + " of the predictions were in the top 5%")
print("rmse error is: " + str(total_rmse), flush=True)
