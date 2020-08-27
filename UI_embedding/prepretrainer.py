import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import tqdm

from UI2Vec import UI2Vec, HiddenLabelPredictorModel
from dataset.vocab import BertScreenVocab

class UI2VecTrainer:
    """
    """

    def __init__(self, predictor: HiddenLabelPredictorModel, dataloader_train, dataloader_test, 
                vocab: BertScreenVocab, vocab_size:int, l_rate: float, n: int, cos_loss: int, bert_size=768):
        """
        """
        self.predictor = predictor
        self.optimizer = Adam(self.predictor.parameters(), lr=l_rate)
        self.vocab = vocab
        self.train_data = dataloader_train
        self.test_data = dataloader_test
        self.vocab_size = vocab_size
        self.cosine_loss = cos_loss
        self.loss = nn.CrossEntropyLoss(reduction='sum')

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
        :return: loss
        """
        total_loss = 0
        total_data = 0

        str_code = "train" if train else "test"

        data_itr = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        # iterate through data_loader
        vocab_embeddings = self.vocab.embeddings.transpose(0,1)
        vocab_embeddings = vocab_embeddings.cuda()
        for idx,data in data_itr:
            element = data[0]
            context = data[1]
            total_data+=len(element[0])
            # forward the training stuff (prediction)
            prediction_output = self.predictor(context) #input here
            element_target_index = self.vocab.get_index(element[0])
            correct_class = element[1]
            # calculate loss for all prediction stuff
            text_prediction_output = torch.narrow(prediction_output, 1, 0, 768)
            class_prediction_output = torch.narrow(prediction_output, 1, 768, prediction_output.size()[1] - 768)
            text_prediction_output = text_prediction_output.cuda()
            class_prediction_output = class_prediction_output.cuda()
            
            classes = torch.arange(self.predictor.num_classes, dtype=torch.long)
            class_comparison = self.predictor.model.embedder.UI_embedder(classes).transpose(0,1).cuda()

            text_dot_products = torch.mm(text_prediction_output, vocab_embeddings)
            class_dot_products = torch.mm(class_prediction_output, class_comparison)
            text_dot_products = text_dot_products.cpu()
            class_dot_products = class_dot_products.cpu()
            prediction_loss = self.loss(text_dot_products, element_target_index)
            prediction_loss+= self.loss(class_dot_products, correct_class)
            total_loss+=float(prediction_loss)
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
        torch.save(self.predictor.state_dict(), output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
