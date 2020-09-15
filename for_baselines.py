import argparse
import json
import numpy as np
import tqdm
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
from sentence_transformers import SentenceTransformer
from prediction import BaselinePredictor
from UI_embedding.plotter import plot_loss


class BaselineDataset(Dataset):
    '''
    has traces, which have screens
    '''
    def __init__(self, embeddings, n):
        self.n = n
        self.traces = self.load_traces(embeddings)
        self.trace_loc_index = []
        self.embeddings = []
        self.load_indices()

    def __getitem__(self, index):
        indexed_trace = self.traces[index]
        # not added unless there are at least n screens in the trace
        traces = []
        if len(indexed_trace) >= self.n:
            starting_index = random.randint(0, len(indexed_trace)-self.n)
            screens = indexed_trace[starting_index:starting_index+self.n-2]
            target_index = self.get_overall_index(index, starting_index+self.n -1)
        return torch.tensor(screens), torch.tensor(target_index)
    
    def __len__(self):
        return len(self.traces)

    def load_traces(self,emb):
        traces = []
        for trace in emb:
            if len(trace) >= self.n:
                traces.append(trace)
        return traces

    def load_indices(self):
        overall_index = 0
        for trace in self.traces:
            self.embeddings += trace
            self.trace_loc_index.append([overall_index + i for i in range(len(trace))])
            overall_index += len(trace)

    def get_overall_index(self, trace_idx, screen_idx):
        return self.trace_loc_index[trace_idx][screen_idx]


class PredictionTrainer():
    def __init__(self, predictor, comparison, train_data, test_data, criterion, optimizer):
        self.predictor = predictor
        self.comparison = torch.tensor(comparison).transpose(0,1).cuda()
        self.train_data = train_data
        self.test_data = test_data
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, epoch):
            loss = self.iteration(epoch, self.train_data)
            return loss

    def test(self, epoch):
        loss = self.iteration(epoch, self.test_data, train=False)
        return loss

    def iteration(self, epoch, data_loader: iter, train=True):
        """
        loop over the data_loader for training or testing
        if train , backward operation is activated
        also auto save the model every epoch

        :param epoch: index of current epoch 
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        """
        total_loss = 0
        total_data = 0
        str_code = "train" if train else "test"
        data_itr = tqdm.tqdm(enumerate(data_loader), desc="EP_%s:%d" % (str_code, epoch),total=len(data_loader),bar_format="{l_bar}{r_bar}")
        for idx, data in data_itr:
            self.optimizer.zero_grad()
            embeddings, idx = data
            total_data+=len(idx)
            embeddings = embeddings.cuda()
            idx = idx.cuda()
            predictions = self.predictor(embeddings)
            dot_products = torch.mm(predictions, self.comparison)
            loss = self.criterion(dot_products, idx)
            total_loss+= float(loss)
            if train:
                loss.backward()
                self.optimizer.step()
        return total_loss/total_data

    def save(self, epoch, file_path="output/trained.model"):
        """
        Saving the current model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.predictor.state_dict(), output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data", required=True, type=str, default=None, help="prefix of precomputed embeddings to test/train predictions")
parser.add_argument("-o", "--output_path", required=True, type=str, help="where to store model")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="traces in a batch")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-n", "--num_predictors", type=int, default=10, help="number of other labels used to predict one")
parser.add_argument("-r", "--rate", type=float, default=0.001, help="learning rate")


args = parser.parse_args()

with open(args.data) as f:
    embeddings = json.load(f, encoding='utf-8')
print(len(embeddings))

dataset = BaselineDataset(embeddings, args.num_predictors)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(val_indices)
train_data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
test_data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)

# generate models
predictor = BaselinePredictor(len(dataset.embeddings[0]))
predictor.cuda()

criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = Adam(predictor.parameters(), lr=args.rate)
trainer = PredictionTrainer(predictor, dataset.embeddings, train_data_loader, test_data_loader, criterion, optimizer)

# training occurs below
test_loss_data = []
train_loss_data = []
for epoch in tqdm.tqdm(range(args.epochs)):
    print("--------")
    print(str(epoch) + " train loss:")
    train_loss = trainer.train(epoch)
    print(train_loss)
    print("--------")
    train_loss_data.append(train_loss)
    if test_data_loader is not None:
        print("--------")
        print(str(epoch) + " test loss:")
        test_loss = trainer.test(epoch)
        print(test_loss)
        print("--------")
        test_loss_data.append(test_loss)
    if (epoch%20)==0:
        print("saved on epoch " + str(epoch))
        trainer.save(epoch, args.output_path)
plot_loss(train_loss_data, test_loss_data, args.output_path)
trainer.save(args.epochs, args.output_path)

