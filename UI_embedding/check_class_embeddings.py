from sentence_transformers import SentenceTransformer
import json
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import os
from dataset.playstore_scraper import get_app_description
from dataset.rico_utils import get_all_texts_from_rico_screen, get_all_labeled_texts_from_rico_screen, ScreenInfo
from dataset.rico_dao import load_rico_screen_dict
from UI2Vec import HiddenLabelPredictorModel, UI2Vec, UIEmbedder


parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", required=True, type=str, help="path where pretrained part was stored")
parser.add_argument("-nc", "--num_clusters", required=True, type=int, help="number of clusters")

args = parser.parse_args()

bert = SentenceTransformer('bert-base-nli-mean-tokens')
bert_size = 768

loaded_model = HiddenLabelPredictorModel(bert, bert_size, 4)
loaded_model.load_state_dict(torch.load(args.model))

class_embeddings = []

for class_name in range(24):
    class_embedding = loaded_model.model.embedder.UI_embedder(torch.tensor(class_name))
    class_embeddings.append(class_embedding)

print(class_embeddings)

from sklearn.cluster import KMeans

num_clusters = args.num_clusters
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(torch.stack(class_embeddings).detach())
assignment = clustering_model.labels_

for cl_no in range(num_clusters):
    clustered_words = [idx for idx in range(len(assignment)) if assignment[idx] == cl_no ]
    print(cl_no)
    print(clustered_words)
