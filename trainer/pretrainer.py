
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import Screen2Vec

class Screen2VecTrainer:
    """
    """

    def __init__(self, model: Screen2Vec, ):
        """
        """
        pass

    def iteration(self, epoch, data_loader, train=True)
        """
        loop over the data_loader for training or testing
        if train , backward operation is activated
        also auto save the model every epoch

        :param epoch: index of current epoch 
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
