
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import tqdm

from .UI2Vec import UI2Vec
from .prediction import HiddenLabelPredictorModel
from .dataset.vocab import BertScreenVocab

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
        if self.cosine_loss:
            self.loss = nn.CosineEmbeddingLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

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
        total_batches = 0

        str_code = "train" if train else "test"

        data_itr = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        # iterate through data_loader
        if not self.cosine_loss:
            vocab_embeddings = self.vocab.embeddings.transpose(0,1)
            vocab_embeddings = vocab_embeddings.cuda()
            for idx,data in data_itr:
                self.optimizer.zero_grad()
                total_batches+=1
                element = data[0]
                context = data[1]
                # forward the training stuff (prediction)
                prediction_output = self.predictor(context) #input here
                element_target_index = self.vocab.get_index(element[0])
                # calculate loss for all prediction stuff
                prediction_output = prediction_output.cuda()
                dot_products = torch.mm(prediction_output, vocab_embeddings)
                dot_products = dot_products.cpu()
                prediction_loss = self.loss(dot_products, element_target_index)
                total_loss+=float(prediction_loss)
                if train:
                    self.optimizer.zero_grad()
                    prediction_loss.backward()
                    self.optimizer.step()
        else:
            for idx,data in data_itr:
                self.optimizer.zero_grad()
                total_batches+=1
                element = data[0]
                context = data[1]
                # forward the training stuff (prediction)
                prediction_output = self.predictor(context) #input here
                element_target_index = self.vocab.get_index(element[0])
                # calculate loss for all prediction stuff
                for i in range(self.vocab_size):
                    ones_vec = -torch.ones(len(prediction_output))
                    for batch in range(len(element_target_index)):
                        if element_target_index[batch] == i:
                            ones_vec[batch] = 1
                    vocab_embedding = self.vocab.get_embedding_for_cosine(i)
                    vocab_embedding = vocab_embedding.repeat(len(prediction_output),1)
                    prediction_loss= self.loss(prediction_output, vocab_embedding, ones_vec)
                    total_loss+=float(prediction_loss)

            # if in train, backwards and optimization
                if train:
                    self.optimizer.zero_grad()
                    prediction_loss.backward()
                    self.optimizer.step()
        return total_loss/total_batches

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
