
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from UI2Vec import UI2Vec
from prediction import HiddenLabelPredictorModel
from dataset.vocab import BertScreenVocab

class UI2VecTrainer:
    """
    """

    def __init__(self, embedder: UI2Vec, dataloader_train, dataloader_test, 
                vocab: BertScreenVocab, vocab_size:int, l_rate: float, n: int, bert_size=768):
        """
        """
        self.loss = nn.CrossEntropyLoss()
        self.UI2Vec = embedder
        self.model = HiddenLabelPredictorModel(embedder, bert_size*n, bert_size, vocab_size) 
        self.optimizer = Adam(self.model.parameters())
        self.vocab = vocab


    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.train_data, train=False)

    def iteration(self, epoch, data_loader: iter, train=True):
        """
        loop over the data_loader for training or testing
        if train , backward operation is activated
        also auto save the model every epoch

        :param epoch: index of current epoch 
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        total_loss = 0
        total_data = 0
        # iterate through data_loader
        for data in data_loader:
            total_data+=1
            element = data[0]
            context = data[1]
            # load data properly
            # forward the training stuff (prediction models)
            prediction_output = self.model.forward(context) #input here

            element_target = vocab.get_index(element[0])
            # calculate NLL loss for all prediction stuff
            prediction_loss = self.loss(prediction_output, element_target)
            # if in train, backwards and optimization
            total_loss+=prediction_loss
            if train:
                self.optimizer.zero_grad()
                prediction_loss.backward()
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
        torch.save(self.UI2Vec.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path