
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from Screen2Vec import Screen2Vec

class Screen2VecTrainer:
    """
    """

    def __init__(self, model: Screen2Vec, dataloader_train, dataloader_test, 
                vocab_size:int, l_rate: float):
        """
        """
        self.criterion = nn.NLLLoss()
        self.model = 
        self.optimizer = Adam(self.model.parameters())
        self.train_data = dataloader_train
        self.test_data = dataloader_test

    def train(self, epoch):
        loss = self.iteration(epoch, self.train_data)
        return loss

    def test(self, epoch):
        loss = self.iteration(epoch, self.test_data, train=False)
        return loss

    def iteration(self, epoch, data_loader: iter, train=True)
        """
        loop over the data_loader for training or testing
        if train , backward operation is activated
        also auto save the model every epoch

        :param epoch: index of current epoch 
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        # iterate through data_loader
        total_loss = 0
        total_batches = 0
        for data in data_loader:
            # load data properly
            UIs, descr, trace_screen_lengths = data
            #TODO; take all but last screen in trace
            # forward the training stuff (prediction models)
            prediction_output = self.model.forward() #input here

            # calculate NLL loss for all prediction stuff
            prediction_loss = self.criterion(prediction_output)
            total_loss+=float(prediction_loss)
            # if in train, backwards and optimization
            if train:
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
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path