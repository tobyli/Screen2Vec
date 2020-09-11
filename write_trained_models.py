import argparse
import torch
import torch.nn as nn
import json
import scipy
import numpy as np
from torch.utils.data import DataLoader
from Screen2Vec import Screen2Vec
from dataset.dataset import PrecompRicoDataset, RicoTrace, RicoScreen
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

dataset = PrecompRicoDataset(args.num_predictors, uis, ui_emb, descr, descr_emb, layouts, args.net_version, True, screen_names)       

vocab = ScreenVocab(dataset)


del ui_emb

end_index = 0
# if args.net_version not in [5]:
#     comp = torch.empty(0,bert_size)
# else:
#     comp = torch.empty(0,bert_size *2)


# while end_index != -1:
#     vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts, vocab_indx_map, vocab_rvs_indx, end_index = vocab.get_all_screens(end_index, 1024)
#     comp_part = predictor.model(vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts, False).squeeze(0)
#     comp = torch.cat((comp, comp_part), dim = 0)

# comp = comp.detach().numpy()

# comp_dict = {}


# for emb_idx in range(len(comp)):
#     names = vocab.get_name(emb_idx)
#     name = "/".join(names.split("/")[-4:])
#     comp_dict[name] = comp[emb_idx].tolist()

# mistakes = []

# with open('model' + str(args.net_version) + 'full.json', 'w', encoding='utf-8') as f:
#     json.dump(comp_dict, f, indent=4)

if args.net_version in [4,6,7,8]:
    end_index = 0
    #comp = torch.empty(0,bert_size*2)
    comp_dict = {}
    while end_index != -1:
        start_index = end_index
        vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts , vocab_indx_map, vocab_rvs_indx, end_index = vocab.get_all_screens(end_index, 1024)
        comp_part = predictor.model(vocab_UIs, vocab_descr, vocab_trace_screen_lengths, vocab_layouts).squeeze(0)
        embeddings = torch.cat((comp_part, vocab_descr.squeeze(0)), dim=1).detach()
        #comp = torch.cat((comp, embeddings), dim = 0)
        for emb_idx in range(len(embeddings)):
            idx = emb_idx + start_index
            names = vocab.get_name(idx)
            name = "/".join(names.split("/")[-4:])
            comp_dict[name] = embeddings[emb_idx].detach().tolist()

    # comp = comp.detach().numpy()


    # print("stop")
    # comp_dict = {}
    # del dataset
    # del vocab

    # for emb_idx in range(len(comp)):
    #     names = vocab.get_name(emb_idx)
    #     name = "/".join(names.split("/")[-4:])
    #     comp_dict[name] = comp[emb_idx].tolist()

    with open('model' + str(args.net_version) + 'descr.json', 'w', encoding='utf-8') as f:
        json.dump(comp_dict, f, indent=4)
