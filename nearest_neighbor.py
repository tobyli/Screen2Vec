import argparse
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
import scipy
import random
import numpy as np
from torch.utils.data import DataLoader
from Screen2Vec import Screen2Vec
from pretrainer import Screen2VecTrainer
from dataset.dataset import RicoDataset, RicoTrace, RicoScreen
from sentence_transformers import SentenceTransformer
from prediction import TracePredictor
from vocab import ScreenVocab


parser = argparse.ArgumentParser()

parser.add_argument("-e", "--emb_path", type=str, default="", help="path to stored embeddings")
parser.add_argument("-s", "--selected", type=list, default=[], help="path to stored embeddings")
parser.add_argument("-v", "--net_version", type=int, default=0, help="0 for regular, 1 to embed location in UIs, 2 to use layout embedding, and 3 to use both")
args = parser.parse_args()

with open("model" + str(args.net_version) + ".json") as f:
    embeddings = json.load(f, encoding='utf-8')

def get_most_relevant_embeddings(src_id, rico_id_embedding_dict: dict, n: int):
    src_embedding = rico_id_embedding_dict[src_id]
    screen_info_similarity_list = []
    for rico_id, embedding in rico_id_embedding_dict.items():
        if (embedding is None or src_embedding is None):
            continue
        if ((isinstance(embedding, int)) and embedding == 0):
            continue
        if ((isinstance(src_embedding, int)) and src_embedding == 0):
            continue
        entry = {}
        entry['rico_id'] = rico_id
        entry['score'] = scipy.spatial.distance.cosine(src_embedding, embedding)
        screen_info_similarity_list.append(entry)
    screen_info_similarity_list.sort(key=lambda x: x['score'])
    return screen_info_similarity_list[1:n+1]

def vector_compose(screen1, screen2, screen3, emb_dict):
    """
    screen1 + screen2 - screen3
    """
    result = emb_dict[screen1] + emb_dict[screen2] - emb_dict[screen3]
    closest = float('inf')
    close_screen = ''
    for id, embedding in emb_dict.items():
        if (embedding is None):
            continue
        if ((isinstance(embedding, int)) and embedding == 0):
            continue
        dist = scipy.spatial.distance.cosine(result, embedding)
        if dist<closest:
            closest = dist
            close_screen = id
    return close_screen

if args.selected:
    for screen in args.selected:
        print(get_most_relevant_embeddings(screen, embeddings, 5))
else:
    n = 0
    keys = list(embeddings.keys())
    while n < 10:
        n+=1
        key = random.choice(keys)
        print("___________________________________")
        print(get_most_relevant_embeddings(key, embeddings, 5))