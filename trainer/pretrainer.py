
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import Screen2Vec

class Screen2VecTrainer:
    """
    """

    def __init__(self, model: Screen2Vec, dataloader_test, dataloader_train, 
                vocab_size:int, l_rate: float):
        """
        """
        self.criterion = nn.NLLLoss()
        self.model = model
        self.optimizer = Adam(self.model.parameters())


    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.train_data, train=False)

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
        for data in data_loader:
            # load data properly
            # forward the training stuff (prediction models)
            prediction_output = self.model.forward() #input here

            # calculate NLL loss for all prediction stuff
            prediction_loss = self.criterion(prediction_output)
            # if in train, backwards and optimization
            if train:
                loss.backward()
                self.optimizer.step()

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