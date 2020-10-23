import argparse
import math
import torch
import torch.nn as nn
import json
import scipy
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from Screen2Vec import Screen2Vec
from dataset.dataset import TesterRicoDataset, RicoTrace, RicoScreen
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

parser.add_argument("-m4", "--model4", required=True, type=str, help="path to pretrained model to test")
parser.add_argument("-m6", "--model6", required=True, type=str, help="path to pretrained model to test")
parser.add_argument("-m7", "--model7", required=True, type=str, help="path to pretrained model to test")
parser.add_argument("-m8", "--model8", required=True, type=str, help="path to pretrained model to test")
parser.add_argument("-m9", "--model9", required=True, type=str, help="path to pretrained model to test")
#parser.add_argument("-v", "--net_version", type=int, default=0, help="0 for regular, 1 to embed location in UIs, 2 to use layout embedding, 3 to use both, 4 with both but no description, 5 to use both but not train description")
parser.add_argument("-d", "--data", required=True, type=str, default=None, help="prefix of precomputed data")
parser.add_argument("-n", "--num_predictors", type=int, default=10, help="number of other labels used to predict one")
args = parser.parse_args()





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


with open(args.data + "layout_embeddings.json") as f:
    layouts = json.load(f, encoding='utf-8')


dataset = TesterRicoDataset(args.num_predictors, uis, ui_emb, descr, descr_emb, layouts, 4, True, screen_names)       

vocab = ScreenVocab(dataset)


model_list = [args.model4, args.model6, args.model7, args.model8, args.model9]
version_list = [4,6,7,8,9]

for model_idx in range(len(model_list)):
    correct = 0
    toppointzeroone = 0
    toppointone = 0
    topone = 0
    topfive = 0
    total = 0

    version = version_list[model_idx]
    print(version)
    model = model_list[model_idx]

    bert_size = 768
    if version in [0,2,6,8]:
        adus = 0
    else:
        # case where coordinates are part of UI rnn
        adus = 4
    if version in [0,1,6,7]:
        adss = 0
    else:
        # case where screen layout vec is used
        adss = 64

    dataset.setting = version
    vocab.setting = version
    data_loader = DataLoader(dataset, collate_fn=pad_collate, batch_size=1)
    orig_model = Screen2Vec(bert_size, additional_ui_size=adus, additional_size_screen=adss, net_version=version)
    predictor = TracePredictor(orig_model, version)
    predictor.load_state_dict(torch.load(model))

    end_index = 0
    comp = torch.empty(0,bert_size*2)
    while end_index != -1:
        vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts , vocab_indx_map, vocab_rvs_indx, end_index = vocab.get_all_screens(end_index, 1024)
        comp_part = predictor.model(vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts).squeeze(0)
        embeddings = torch.cat((comp_part, vocab_descr.squeeze(0)), dim=1)
        comp = torch.cat((comp, embeddings), dim = 0)

    comp = comp.detach()


    error = nn.MSELoss()


    data_itr = tqdm.tqdm(enumerate(data_loader),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
    i = 0
    total_se = 0
    total_vector_lengths = 0
    for data_idx, data in data_itr:
        UIs, descr, trace_screen_lengths, index, layouts = data
    # run it through the network
        UIs, descr, trace_screen_lengths, index, layouts = data
        i+=len(index)
        # forward the training stuff (prediction)
        c,result,_ = predictor(UIs, descr, trace_screen_lengths, layouts, False)
        descr = torch.narrow(descr,1,0,1).squeeze(1)
        c = torch.cat((c,descr),dim=-1)
        distances = scipy.spatial.distance.cdist(c.detach(), comp, "cosine")
        # find which vocab vector has the smallest cosine distance
        for idx in range(len(index)):
            correct_index = vocab_rvs_indx[index[idx][0]][index[idx][1]]
            #print(c.size())
            #print(comp[correct_index].size())
            diff = c.detach()[idx]-comp[correct_index]
            sqer = sum(diff**2)
            total_se += sqer
            total_vector_lengths += np.linalg.norm(comp[correct_index])
            
            comp_len = len(distances[idx])

            temp = np.argpartition(distances[idx], (0,int(0.0001 * comp_len), int(0.001 * comp_len), int(0.01 * comp_len), int(0.05 * comp_len), int(0.1 * comp_len)))
            closest_idx = temp[0]
            closest_pointzerooneperc = temp[:int(0.0001 * comp_len)]
            closest_pointoneperc = temp[:int(0.001 * comp_len)]
            closest_oneperc = temp[:int(0.01 * comp_len)]
            closest_fiveperc = temp[:int(0.05 * comp_len)]

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
    print("rmse error is: " + str(total_rmse), flush=True)

