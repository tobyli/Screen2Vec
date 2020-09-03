import argparse
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
import scipy
import numpy as np
from torch.utils.data import DataLoader
from UI2Vec import UI2Vec, HiddenLabelPredictorModel
from prepretrainer import UI2VecTrainer
from dataset.dataset import RicoDataset, RicoScreen, ScreenDataset
from dataset.vocab import BertScreenVocab


parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", required=True, type=str, help="path to pretrained model to test")
parser.add_argument("-n", "--num_predictors", type=int, default=10, help="number of other labels used to predict one when model was trained")
parser.add_argument("-r", "--range", type=float, default=0.1, help="what proportion of results to look in")
parser.add_argument("-v", "--vocab_path", required=True, type=str, help="path to json of text in vocab")
parser.add_argument("-x", "--extra", type=int, default=0, help="1 to display clustering results")
args = parser.parse_args()

n = args.num_predictors
bert_size = 768
bert = bert = SentenceTransformer('bert-base-nli-mean-tokens')
predictor = HiddenLabelPredictorModel(bert, bert_size, n)
predictor.load_state_dict(torch.load(args.model))

vocab_path = args.vocab_path

with open(vocab_path) as f:
    vocab_list = json.load(f, encoding='utf8')

vocab = BertScreenVocab(vocab_list, len(vocab_list), bert)

print("length of vocab is " + str(len(vocab.embeddings)))
print("length of vocab is " + str(len(vocab.vocab_list)))

input_path = 'dataset/data/'


print(int(args.range * len(vocab_list)))
correct_text = 0
correct_class = 0
correct_both = 0
total_text = 0
total_class = 0
total_both = 0
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
    prediction_output = predictor(context).cpu() #input here
    element_target_index = vocab.get_index(element[0])
    target_class = element[1]

    text_prediction_output = torch.narrow(prediction_output, 1, 0, 768)
    class_prediction_output = torch.narrow(prediction_output, 1, 768, prediction_output.size()[1] - 768).detach()

    # find which vocab vector has the smallest cosine distance
    text_distances = scipy.spatial.distance.cdist(text_prediction_output.detach().numpy(), vocab.embeddings, "cosine")[0]

    text_temp = np.argpartition(text_distances, (1,int(args.range * len(vocab_list))))
    text_closest_idx = text_temp[:int(args.range * len(vocab_list))]

    classes = torch.arange(predictor.num_classes, dtype=torch.long)
    class_comparison = predictor.model.embedder.UI_embedder(classes).detach()
    class_distances = scipy.spatial.distance.cdist(class_prediction_output, class_comparison, "cosine")[0]

    class_temp = np.argpartition(class_distances, (1,int(args.range * len(class_prediction_output))))
    class_closest_idx = class_temp[:int(args.range * len(class_comparison))]

    if int(element_target_index) is not 0:
        total_text+=1
        if int(element_target_index) in text_closest_idx:
            correct_text +=1
    if int(target_class) is not 0:
        total_class +=1
        if int(target_class) in class_closest_idx:
            correct_class +=1
    if int(target_class) is not 0 and int(element_target_index) is not 0:
        total_both +=1
        if int(target_class) in class_closest_idx and int(element_target_index) in text_closest_idx:
            correct_both +=1



print(correct_text/total_text)
print(correct_class/total_class)
print(correct_both/total_both)


if args.extra:
    from sklearn.cluster import KMeans

    corpus = [screen.labeled_text for screen in dataset.screens]
    corpus_text = [bundle[0] for text in corpus for bundle in text] 
    corpus_class = torch.tensor([bundle[1] for text in corpus for bundle in text])
    corpus_embeddings = model.model((corpus_text, corpus_class)).detach().numpy()

    num_clusters = 50
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    assignment = clustering_model.labels_

    with open("cluster_output.txt", "w", encoding="utf-8") as f:
        for cl_no in range(num_clusters):
            clustered_words = [corpus_text[idx] + "\n" for idx in range(len(assignment)) if assignment[idx] == cl_no ]
            print(cl_no)
            print(clustered_words[:10])
            f.write(str(cl_no) + ":\n")
            f.writelines(clustered_words)
