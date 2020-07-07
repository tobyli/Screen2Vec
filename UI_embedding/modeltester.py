import argparse
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
import scipy
import numpy as np
from torch.utils.data import DataLoader
from UI_embedding.UI2Vec import UI2Vec
from UI_embedding.prediction import HiddenLabelPredictorModel
from UI_embedding.prepretrainer import UI2VecTrainer
from UI_embedding.dataset.dataset import RicoDataset, RicoScreen, ScreenDataset
from UI_embedding.dataset.vocab import BertScreenVocab


parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", required=True, type=str, help="path to pretrained model to test")
parser.add_argument("-n", "--num_predictors", type=int, default=10, help="number of other labels used to predict one when model was trained")
parser.add_argument("-r", "--range", type=float, default=0.1, help="what proportion of results to look in")
parser.add_argument("-v", "--vocab_path", required=True, type=str, help="path to json of text in vocab")
parser.add_argument("-x", "--extra", type=int, default=0, help="1 to display some extra results")
args = parser.parse_args()

n = args.num_predictors
bert_size = 768
bert = bert = SentenceTransformer('bert-base-nli-mean-tokens')
model = HiddenLabelPredictorModel(bert, bert_size, n)
model.load_state_dict(torch.load(args.model))

vocab_path = args.vocab_path

with open(vocab_path) as f:
    vocab_list = json.load(f, encoding='utf8')

vocab = BertScreenVocab(vocab_list, len(vocab_list), bert)

print("length of vocab is " + str(len(vocab.embeddings)))
print("length of vocab is " + str(len(vocab.vocab_list)))

input_path = 'dataset/data/'


print(int(args.range * len(vocab_list)))
correct = 0
total = 0
# load the data
dataset_rico = RicoDataset(input_path)
dataset = ScreenDataset(dataset_rico, n)
data_loader = DataLoader(dataset, batch_size=1)


i = 0
for data in data_loader:
# run it through the network
    element = data[0]
    context = data[1]
    # forward the training stuff (prediction)
    prediction_output = model(context) #input here
    element_target_index = vocab.get_index(element[0])

    # find which vocab vector has the smallest cosine distance
    distances = scipy.spatial.distance.cdist(prediction_output.detach().numpy(), vocab.embeddings, "cosine")[0]

    temp = np.argpartition(-distances, int(args.range * len(vocab_list)))
    closest_idx = temp[:int(args.range * len(vocab_list))]

    if int(element_target_index) in closest_idx:
        correct +=1
    total+=1

    if (i<100):
        i+=1
        print("intended: " + vocab.get_text(element_target_index))

        print("predicted: " + vocab.get_text(closest_idx[0]))

print(correct/total)
