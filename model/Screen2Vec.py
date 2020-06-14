import torch.nn as nn


class Screen2Vec(nn.Module):

    """
    Model intended to semantically embed the content of UI screens into vectors
    """

    def __init__(self, bert_size, num_classes=24):
        """
        describe params here
        """
        super(self, Screen2Vec)
        self.net = nn.RNN(bert_size, bert_size)

    def forward(self, params):
        """
        describe params here
        """
        pass