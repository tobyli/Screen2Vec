import torch
import torch.nn as nn
from UI2Vec import UI2Vec

class HiddenLabelPredictorModel(nn.Module):
    """
    combines the n closest UI elements (text plus class) to predict the embedding
    of a different one on the same screen
    """
    def __init__(self, bert, bert_size, n):
        super().__init__()
        self.lin = nn.Linear(bert_size*n, bert_size)
        self.model = UI2Vec(bert)
        self.n = n
        self.bert_size = bert_size

    def forward(self, context):
        # add all of the embedded texts into a megatensor
        # if missing (less than n)- add padding
        text_embedding = self.model(context[0])
        for index in range(1, self.n):
            to_add = self.model(context[index])
            text_embedding = torch.cat((text_embedding, to_add),1)
        return self.lin(text_embedding)