import argparse
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
import math
import scipy
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from UI2Vec import UI2Vec, HiddenLabelPredictorModel
from prepretrainer import UI2VecTrainer
from dataset.dataset import RicoDataset, RicoScreen, ScreenDataset
from dataset.vocab import BertScreenVocab

# tests UI models for prediction accuracy

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", required=True, type=str, help="path to pretrained model to test")
parser.add_argument("-n", "--num_predictors", type=int, default=16, help="number of other labels used to predict one when model was trained")
parser.add_argument("-v", "--vocab_path", required=True, type=str, help="path to json of text in vocab")
parser.add_argument("-ve", "--vocab_embedding_path", type=str, help="path to vocab embedding")
parser.add_argument("-d", "--data", required=True, type=str, default=None, help="path to dataset")
parser.add_argument("-hi", "--hierarchy", action="store_true")
args = parser.parse_args()

n = args.num_predictors
bert_size = 768
bert = bert = SentenceTransformer('bert-base-nli-mean-tokens')
predictor = HiddenLabelPredictorModel(bert, bert_size, n)
predictor.load_state_dict(torch.load(args.model))

vocab_path = args.vocab_path

with open(vocab_path) as f:
    vocab_list = json.load(f, encoding='utf8')

vocab = BertScreenVocab(vocab_list, len(vocab_list), bert, embedding_path=args.vocab_embedding_path)


input_path = args.data


correct_text = 0
correct_class = 0
correct_both = 0
total_text = 0
total_class = 0
total_both = 0
toppointzeroone_text = 0
toppointone_text = 0
topone_text = 0
topfive_text = 0
topten_text = 0
toppointzeroone_class = 0
toppointone_class = 0
topone_class = 0
topfive_class = 0
topten_class = 0
toppointzeroone_both = 0
toppointone_both = 0
topone_both = 0
topfive_both = 0
topten_both = 0

total_se = 0
total_vector_lengths = 0
# load the data
dataset_rico = RicoDataset(input_path, hierarchy=args.hierarchy)
dataset = ScreenDataset(dataset_rico, n)
data_loader = DataLoader(dataset, batch_size=1)


i = 0

data_itr = tqdm.tqdm(enumerate(data_loader),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

for idx, data in data_itr:
# run it through the network
    i+=1
    element = data[0]
    context = data[1]
    # forward the training stuff (prediction)
    prediction_output = predictor(context).cpu() #input here
    element_target_index = vocab.get_index(element[0])
    target_class = element[1]

    classes = torch.arange(predictor.num_classes, dtype=torch.long)
    class_comparison = predictor.model.embedder.UI_embedder(classes).detach()

    diff = (prediction_output - torch.cat((vocab.embeddings[int(element_target_index)], class_comparison[int(target_class)]))).detach().squeeze(0)
    total_se += sum(diff**2)
    total_vector_lengths += np.linalg.norm(diff)
    
    text_prediction_output = torch.narrow(prediction_output, 1, 0, 768)
    class_prediction_output = torch.narrow(prediction_output, 1, 768, prediction_output.size()[1] - 768).detach()

    # find which vocab vector has the smallest cosine distance
    text_distances = scipy.spatial.distance.cdist(text_prediction_output.detach().numpy(), vocab.embeddings, "cosine")[0]

    text_temp = np.argpartition(text_distances, (1, int(0.0001 * len(text_distances)),int(0.001 * len(text_distances)),int(0.01 * len(text_distances)),int(0.05 * len(text_distances)),int(0.1 * len(text_distances))))
    text_closest_idx = text_temp[0]
    text_closest_pointzerooneperc = text_temp[:int(0.0001 * len(text_distances))]
    text_closest_pointoneperc = text_temp[:int(0.001 * len(text_distances))]
    text_closest_oneperc = text_temp[:int(0.01 * len(text_distances))]
    text_closest_fiveperc = text_temp[:int(0.05 * len(text_distances))]
    text_closest_tenperc = text_temp[:int(0.1 * len(text_distances))]

    class_distances = scipy.spatial.distance.cdist(class_prediction_output, class_comparison, "cosine")[0]

    class_temp = np.argpartition(class_distances, (1,int(0.0001 * len(class_distances)),int(0.001 * len(class_distances)),int(0.01 * len(class_distances)),int(0.05 * len(class_distances)),int(0.1 * len(class_distances))))
    class_closest_idx = class_temp[0]
    class_closest_pointzerooneperc = class_temp[:int(0.0001 * len(class_distances))]
    class_closest_pointoneperc = class_temp[:int(0.001 * len(class_distances))]
    class_closest_oneperc = class_temp[:int(0.01 * len(class_distances))]
    class_closest_fiveperc = class_temp[:int(0.05 * len(class_distances))]
    class_closest_tenperc = class_temp[:int(0.1 * len(class_distances))]

    if int(element_target_index) is not 0:
        total_text+=1
        if int(element_target_index)==text_closest_idx:
            correct_text +=1
            toppointzeroone_text+=1
            toppointone_text +=1
            topone_text +=1
            topfive_text +=1
            topten_text +=1
        elif int(element_target_index) in text_closest_pointzerooneperc:
            toppointzeroone_text +=1
            toppointone_text +=1
            topone_text +=1
            topfive_text +=1
            topten_text +=1
        elif int(element_target_index) in text_closest_pointoneperc:
            toppointone_text +=1
            topone_text +=1
            topfive_text +=1
            topten_text +=1
        elif int(element_target_index) in text_closest_oneperc:
            topone_text +=1
            topfive_text +=1
            topten_text +=1
        elif int(element_target_index) in text_closest_fiveperc:
            topfive_text +=1
            topten_text +=1
        elif int(element_target_index) in text_closest_tenperc:
            topten_text +=1
    if int(target_class) is not 0:
        total_class+=1
        if int(target_class)==class_closest_idx:
            correct_class +=1
            toppointzeroone_class+=1
            toppointone_class +=1
            topone_class +=1
            topfive_class +=1
            topten_class +=1
        elif int(target_class) in class_closest_pointzerooneperc:
            toppointzeroone_class +=1
            toppointone_class +=1
            topone_class +=1
            topfive_class +=1
            topten_class +=1
        elif int(target_class) in class_closest_pointoneperc:
            toppointone_class +=1
            topone_class +=1
            topfive_class +=1
            topten_class +=1
        elif int(target_class) in class_closest_oneperc:
            topone_class +=1
            topfive_class +=1
            topten_class +=1
        elif int(target_class) in class_closest_fiveperc:
            topfive_class +=1
            topten_class +=1
        elif int(target_class) in class_closest_tenperc:
            topten_class +=1
    if int(target_class) is not 0 and int(element_target_index) is not 0:
        total_both +=1
        if int(target_class)==class_closest_idx and int(element_target_index)==text_closest_idx:
            correct_both +=1
            toppointzeroone_both+=1
            toppointone_both +=1
            topone_both +=1
            topfive_both +=1
            topten_both +=1
        elif int(target_class) in class_closest_pointzerooneperc and int(element_target_index) in text_closest_pointzerooneperc:
            toppointzeroone_both +=1
            toppointone_both +=1
            topone_both +=1
            topfive_both +=1
            topten_both +=1
        elif int(target_class) in class_closest_pointoneperc and int(element_target_index) in text_closest_pointoneperc:
            toppointone_both +=1
            topone_both +=1
            topfive_both +=1
            topten_both +=1
        elif int(target_class) in class_closest_oneperc and int(element_target_index) in text_closest_oneperc:
            topone_both +=1
            topfive_both +=1
            topten_both +=1
        elif int(target_class) in class_closest_fiveperc and int(element_target_index) in text_closest_fiveperc:
            topfive_both +=1
            topten_both +=1
        elif int(target_class) in class_closest_tenperc and int(element_target_index) in text_closest_tenperc:
            topten_both +=1

rmse = math.sqrt(total_se/i)/(total_vector_lengths/i)
print(str(correct_text/total_text) + " of the text predictions were exactly correct")
print(str(toppointzeroone_text/total_text) + " of the text predictions were in the top 0.01%")
print(str(toppointone_text/total_text) + " of the text predictions were in the top 0.1%")
print(str(topone_text/total_text) + " of the text predictions were in the top 1%")
print(str(topfive_text/total_text) + " of the text predictions were in the top 5%")
print(str(topten_text/total_text) + " of the text predictions were in the top 10%")

print(str(correct_class/total_class) + " of the class predictions were exactly correct")
print(str(toppointzeroone_class/total_class) + " of the class predictions were in the top 0.01%")
print(str(toppointone_class/total_class) + " of the class predictions were in the top 0.1%")
print(str(topone_class/total_class) + " of the class predictions were in the top 1%")
print(str(topfive_class/total_class) + " of the class predictions were in the top 5%")
print(str(topten_class/total_class) + " of the class predictions were in the top 10%")

print(str(correct_both/total_both) + " of the predictions were right on both counts")
print(str(toppointzeroone_both/total_both) + " of the predictions were in the top 0.01% on both counts")
print(str(toppointone_both/total_both) + " of the predictions were in the top 0.1% on both counts")
print(str(topone_both/total_both) + " of the predictions were in the top 1% on both counts")
print(str(topfive_both/total_both) + " of the predictions were in the top 5% on both counts")
print(str(topten_both/total_both) + " of the predictions were in the top 10% on both counts")

print("rmse error is: " + str(rmse))
