import torch
import torch.nn as nn
from Screen2Vec import Screen2Vec

class TracePredictor(nn.Module):
    def __init__(self, embedding_model: Screen2Vec):
        super().__init__()
        self.model = embedding_model
        self.bert_size = self.model.bert_size
        self.combiner = nn.rnn(self.bert_size, self.bert_size, batch_first=True)

    def forward(self, context):
        # process context TODO: make this correct
        UIs, descr, trace_screen_lengths = context
        self.model(context)
        # create initial hidden state
        h = torch.zeros(self.bert_size, batch_size)
        # run through model
        input = torch.nn.utils.rnn.pack_padded_sequence(context, orig_lengths, batch_first=True)
        output, h = self.combiner(input, h)
        output, orig_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output