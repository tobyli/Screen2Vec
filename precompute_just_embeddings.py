from sentence_transformers import SentenceTransformer
import json
import numpy as np
import argparse
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
import os
from dataset.playstore_scraper import get_app_description
from dataset.rico_utils import get_all_texts_from_rico_screen, get_all_labeled_texts_from_rico_screen, ScreenInfo
from dataset.rico_dao import load_rico_screen_dict
from UI_embedding.prediction import HiddenLabelPredictorModel
from dataset.dataset_old import RicoDataset, RicoTrace, RicoScreen

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

parser.add_argument("-m", "--model", required=True, type=str, help="path where pretrained part was stored")
parser.add_argument("-p", "--prefix", required=True, type=str, help="prefix to output files")

args = parser.parse_args()

bert = SentenceTransformer('bert-base-nli-mean-tokens')
bert_size = 768

loaded_model = HiddenLabelPredictorModel(bert, bert_size, 4)
loaded_model.load_state_dict(torch.load(args.model))

UI_embedding = []

with open(args.prefix + 'uis.json') as f:
    UIs = json.load(f, encoding='utf-8')

with open(args.prefix + 'lengths.json') as f:
    screen_lengths = json.load(f, encoding='utf-8')


flat_screens = [u for trace in UIs for screen in trace for u in screen]

ui_data = ScreensList(flat_screens)
ui_loader = DataLoader(ui_data, batch_size=len(flat_screens))
embedding = torch.empty(len(flat_screens), 768)
for data in ui_loader:
    embedding = loaded_model.model(data).detach()


print(embedding.size())
screen_index = 0
for trace in tqdm.tqdm(range(len(screen_lengths))):
    trace_emb = []
    for screen in range(len(screen_lengths[trace])):
        screen_emb = embedding[screen_index:screen_index+screen_lengths[trace][screen]]
        screen_index+= screen_lengths[trace][screen]
        screen_emb = screen_emb.tolist()
        trace_emb.append(screen_emb)
    UI_embedding.append(trace_emb)


with open(args.prefix + 'ui_emb.json', 'w', encoding='utf-8') as f:
    json.dump(UI_embedding, f, indent=4)
