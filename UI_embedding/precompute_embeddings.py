from sentence_transformers import SentenceTransformer
import json
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
import os
from dataset.playstore_scraper import get_app_description
from dataset.rico_utils import get_all_texts_from_rico_screen, get_all_labeled_uis_from_rico_screen, ScreenInfo
from dataset.rico_dao import load_rico_screen_dict
from UI2Vec import HiddenLabelPredictorModel

# file to precompute the UI embeddings for later use in Screen model training

class ScreensList(Dataset):
    """
    Just a list of screens so that dataloader can be used
    """
    def __init__(self, ui_list):
        self.uis = ui_list
    
    def __getitem__(self, index):
        return self.uis[index]
    
    def __len__(self):
        return len(self.uis)

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True, type=str, help="dataset to precompute embeddings for")
parser.add_argument("-m", "--model", required=True, type=str, help="path where pretrained part was stored")
parser.add_argument("-p", "--prefix", required=True, type=str, help="prefix to output files")
parser.add_argument("-n", "--n", default=16, type=int, help="number of predictors used")

args = parser.parse_args()

bert = SentenceTransformer('bert-base-nli-mean-tokens')
bert_size = 768

loaded_model = HiddenLabelPredictorModel(bert, bert_size, args.n)
loaded_model.load_state_dict(torch.load(args.model))

descriptions = []
description_embeddings = {}
UIs = []
UI_embedding = []
screen_names = []

trace_to_index = {}


i = 0
for package_dir in os.listdir(args.dataset):
    if os.path.isdir(args.dataset + '/' + package_dir):
        # for each package directory
        package_name = package_dir
        try:
            descr = get_app_description(package_name)
        except TypeError as e:
            descr = ''
            print(str(e) + ': ' + args.dataset)
        for trace_dir in os.listdir(args.dataset + '/' + package_dir):
            if os.path.isdir(args.dataset + '/' + package_dir + '/' + trace_dir) and (not trace_dir.startswith('.')):
                trace_id = package_dir + trace_dir[-1]
                trace_to_index[trace_id] = i
                print(i)
                i+=1
                descriptions.append(descr)
                UIs_trace = []
                screen_names_trace = []
                for view_hierarchy_json in os.listdir(args.dataset + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies'):
                    if view_hierarchy_json.endswith('.json') and (not view_hierarchy_json.startswith('.')):
                        json_file_path = args.dataset + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies' + '/' + view_hierarchy_json
                        screen_names_trace.append(json_file_path)
                        try:
                            with open(json_file_path) as f:
                                rico_screen = load_rico_screen_dict(json.load(f))
                                labeled_text = get_all_labeled_uis_from_rico_screen(rico_screen)
                        except TypeError as e:
                            print(str(e) + ': ' + args.dataset)
                            labeled_text = []
                        UIs_trace.append(labeled_text)
                UIs.append(UIs_trace)
                screen_names.append(screen_names_trace)

with open(args.prefix + 'descr.json', 'w', encoding='utf-8') as f:
    json.dump(descriptions, f, indent=4)

with open(args.prefix + 'uis.json', 'w', encoding='utf-8') as f:
    json.dump(UIs, f, indent=4)
with open(args.prefix + 'screen_names.json', 'w', encoding='utf-8') as f:
    json.dump(screen_names, f, indent=4)

# screen_lengths = []
# for trace in UIs:
#     trace_screen_lengths = []
#     for screen in trace:
#         trace_screen_lengths.append(len(screen))
#     screen_lengths.append(trace_screen_lengths)

# with open(args.prefix + 'lengths.json', 'w', encoding='utf-8') as f:
#     json.dump(screen_lengths, f, indent=4)

# with open(args.prefix + 'trace_idx.json', 'w', encoding='utf-8') as f:
#     json.dump(trace_to_index, f, indent=4)

description_embeddings = list(bert.encode(descriptions))
np.save(args.prefix + 'dsc_emb', description_embeddings)

screen_lengths = []
for trace in UIs:
    trace_screen_lengths = []
    for screen in trace:
        trace_screen_lengths.append(len(screen))
    screen_lengths.append(trace_screen_lengths)

flat_screens = [u for trace in UIs for screen in trace for u in screen]

num_traces = len(UIs)
parcel_size = int(num_traces/9)

for j in range(10):
    UI_embedding = []
    start_trace = j*parcel_size
    end_trace = min((j+1)*parcel_size, num_traces)
    parcel = UIs[start_trace:end_trace]
    flat_screens = [u for trace in parcel for screen in trace for u in screen]

    ui_data = ScreensList(flat_screens)
    ui_loader = DataLoader(ui_data, batch_size=len(flat_screens))
    embedding = torch.empty(len(flat_screens), 768)
    for data in ui_loader:
        embedding = loaded_model.model(data).detach()


    print(embedding.size())
    screen_index = 0
    for trace in tqdm.tqdm(range(start_trace, end_trace)):
        trace_emb = []
        for screen in range(len(screen_lengths[trace])):
            screen_emb = embedding[screen_index:screen_index+screen_lengths[trace][screen]]
            screen_index+= screen_lengths[trace][screen]
            screen_emb = screen_emb.tolist()
            trace_emb.append(screen_emb)
        UI_embedding.append(trace_emb)

    with open(args.prefix + str(j) + '_ui_emb.json', 'w', encoding='utf-8') as f:
            json.dump(UI_embedding, f, indent=4)
