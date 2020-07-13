import torch
import torch.nn as nn
from Screen2Vec import Screen2Vec

class TracePredictor(nn.Module):
    def __init__(self, embedding_model: Screen2Vec):
        super().__init__()
        self.model = embedding_model
        self.bert_size = self.model.bert_size
        self.combiner = nn.LSTM(self.bert_size, self.bert_size, batch_first=True)

    def forward(self, UIs, descr, trace_screen_lengths):
        # embed all of the screens
        screens = self.model(UIs, descr, trace_screen_lengths)
        #take all but last element of each trace, store as context
        context = torch.narrow(screens, 1, 0, screens.size()[1]-1)
        result = torch.narrow(screens, 1, screens.size()[1]-1, 1).squeeze(1)
        # run through model
        output, (h,c) = self.combiner(context)
        return c[0], result