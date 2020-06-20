import torch
import torch.nn as nn
from UI2Vec import UI2Vec

class HiddenLabelPredictorModel(nn.Module):
    def __init__(self, model: UI2Vec, bert_size, vocab_size, n):
        super.__init__()
        self.lin = nn.Linear(bert_size*n, vocab_size)
        self.model = model
        self.n = n
        self.bert_size = bert_size

    def forward(self, context):
        # add all of the embedded texts into a megatensor
        # if missing (less than n)- add padding
        input_vector = self.model(context[0])
        for index in range(1, len(context)):
            text_embedding = torch.cat((input_vector, self.model(context[index])),0)
        num_zeros = self.n * self.bert_size - len(text_embedding)
        x = torch.cat((text_embedding, torch.zeros(num_zeros)), 0)
        return self.lin(x)