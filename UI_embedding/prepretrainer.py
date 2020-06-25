
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

    def __init__(self, embedder: UI2Vec, predictor: HiddenLabelPredictorModel, dataloader_train, dataloader_test, 
                vocab: BertScreenVocab, vocab_size:int, l_rate: float, n: int, cos_loss: int, bert_size=768):
        """
        """
        self.UI2Vec = embedder
        self.predictor = predictor
        self.optimizer = Adam(self.predictor.parameters())
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
        total_data = 0
        # iterate through data_loader
        for data in data_loader:
            total_data+=1
            element = data[0]
            context = data[1]
            # forward the training stuff (prediction)
            prediction_output = self.predictor.forward(context) #input here
            element_target_index = self.vocab.get_index(element[0])
            # calculate loss for all prediction stuff
            if self.cosine_loss:
                for i in range(self.vocab_size):
                    ones_vec = -torch.ones(len(prediction_output))
                    for batch in range(len(element_target_index)):
                        if element_target_index[batch] == i:
                            ones_vec[batch] = 1
                    vocab_embedding = self.vocab.get_embedding_for_cosine(i, len(prediction_output))
                    prediction_loss= self.loss(prediction_output, vocab_embedding, ones_vec)
                    total_loss+=prediction_loss
            else: 
                vocab_embedding = self.vocab.embeddings.transpose(0,1)
                dot_products = torch.mm(prediction_output, vocab_embedding)
                prediction_loss = self.loss(dot_products, element_target_index)
                total_loss+=prediction_loss
            # if in train, backwards and optimization
            if train:
                self.optimizer.zero_grad()
                prediction_loss.backward()
                self.optimizer.step()
        return total_loss/total_data #TODO think about what this amount means wrt batches

    def save(self, epoch, file_path="output/trained.model"):
        """
        Saving the current model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.predictor.state_dict, output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
