import torch.nn as nn


class Screen2Vec(nn.Module):

    """
    Model intended to semantically embed the content of UI screens into vectors
    """

    def __init__(self, bert_size=768, num_classes=24, dropout=0):
        """
        describe params here
        """
        super(self, Screen2Vec)
        self.net = nn.LSTM(input_size=bert_size, hidden_state=bert_size)

    def forward(self, params):
        """
        describe params here
        """
        # for UI on screen, forward for each of them 