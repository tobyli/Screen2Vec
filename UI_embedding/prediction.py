import torch
import torch.nn as nn
from UI2Vec import UI2Vec

class HiddenLabelPredictorModel(nn.Module):
    def __init__(self, model: UI2Vec, bert, bert_size, n):
        super().__init__()
        self.lin = nn.Linear(bert_size*n, bert_size)
        self.model = model
        self.n = n
        self.bert_size = bert_size

    def forward(self, context):
        # add all of the embedded texts into a megatensor
        # if missing (less than n)- add padding
        #print(context)
        text_embedding = self.model(context[0])
        for index in range(1, len(context)):
            to_add = self.model(context[index])
            text_embedding = torch.cat((text_embedding, to_add),1)
            print(text_embedding.size())
        return self.lin(text_embedding)