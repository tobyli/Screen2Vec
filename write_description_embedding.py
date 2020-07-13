from sentence_transformers import SentenceTransformer
import json
import numpy as np
import argparse
import torch
from dataset.playstore_scraper import get_app_description
from dataset.rico_utils import get_all_texts_from_rico_screen, get_all_labeled_texts_from_rico_screen, ScreenInfo
from dataset.rico_dao import load_rico_screen_dict
from UI_embedding.prediction import HiddenLabelPredictorModel
from dataset.dataset_old import RicoDataset, RicoTrace, RicoScreen

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True, type=str, help="dataset to precompute embeddings for")
parser.add_argument("-m", "--model", required=True, type=str, help="path where pretrained part was stored")

args = parser.parse_args()

bert = SentenceTransformer('bert-base-nli-mean-tokens')
bert_size = 768

loaded_model = HiddenLabelPredictorModel(bert, bert_size, 4)
loaded_model.load_state_dict(torch.load(args.model))

train_dataset = RicoDataset(loaded_model.model, args.dataset, 4)

descriptions = []
description_embeddings = []
UIs = []
UI_embedding = []

for trace in train_dataset.traces:
    d_temp = []
    d_e_temp = []
    u_temp = []
    u_e_temp = []
    for screen in trace.trace_screens:
        d_temp.append(screen.app_description)
        d_e_temp.append([d_e.tolist() for d_e in screen.descr_emb])
        u_temp.append(screen.labeled_text)
        u_e_temp.append([u_e.detach().tolist() for u_e in screen.UI_embeddings])
    descriptions.append(d_temp)
    description_embeddings.append(d_e_temp)
    UIs.append(u_temp)
    UI_embedding.append(u_e_temp)

with open('uis.json', 'w', encoding='utf-8') as f:
    json.dump(UIs, f, indent=4)

with open('ui_emb.json', 'w', encoding='utf-8') as f:
    json.dump(UI_embedding, f, indent=4)

with open('descr.json', 'w', encoding='utf-8') as f:
    json.dump(descriptions, f, indent=4)

with open('dsc_emb.json', 'w', encoding='utf-8') as f:
    json.dump(description_embeddings, f, indent=4)