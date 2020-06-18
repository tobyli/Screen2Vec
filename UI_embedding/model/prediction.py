import torch.nn as nn
from UI2Vec import UI2Vec

class HiddenLabelPredictorModel(nn.Module):
    def __init__(self, model: UI2Vec, bert_size, vocab_size, n):
        super.__init__()
        self.lin = nn.Linear(bert_size*n, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.model

    def forward(self, input_vector):
        x = self.model(input_vector)
        return self.softmax(self.lin(x))