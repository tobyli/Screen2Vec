import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
import scipy
import numpy as np
from torch.utils.data import DataLoader
from UI2Vec import UI2Vec
from prediction import HiddenLabelPredictorModel
from prepretrainer import UI2VecTrainer
from dataset.dataset import RicoDataset, RicoScreen, ScreenDataset
from dataset.vocab import BertScreenVocab

n = 4
bert_size = 768
bert = bert = SentenceTransformer('bert-base-nli-mean-tokens')
model = HiddenLabelPredictorModel(bert, bert_size, n)
model.load_state_dict(torch.load('output_model.ep100'))

vocab_path = 'dataset/vocab_sm.json'

with open(vocab_path) as f:
    vocab_list = json.load(f, encoding='utf8')

vocab = BertScreenVocab(vocab_list, len(vocab_list), bert)

input_path = 'dataset/data/'

correct = 0
total = 0
# load the data
dataset_rico = RicoDataset(input_path)
dataset = ScreenDataset(dataset_rico, n)
data_loader = DataLoader(dataset, batch_size=1)

for data in data_loader:
# run it through the network
    element = data[0]
    context = data[1]
    # forward the training stuff (prediction)
    prediction_output = model.forward(context) #input here
    element_target_index = vocab.get_index(element[0])
    # find which vocab vector has the smallest cosine distance
    distances = scipy.spatial.distance.cdist(prediction_output.detach().numpy(), vocab.embeddings, "cosine")

    temp = np.argpartition(-distances, n*10)
    closest_idx = temp[:(n*10)]
    if element_target_index in closest_idx:
        correct +=1
    total+=1
# find which vocab vector has the largest dot product

print(correct/total)