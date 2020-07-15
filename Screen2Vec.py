import torch
import torch.nn as nn
from UI_embedding.UI2Vec import UI2Vec

class Screen2Vec(nn.Module):

    """
    Model intended to semantically embed the content of UI screens into vectors
    """

    def __init__(self, bert_size=768, num_classes=24):
        """
        describe params here
        """
        super().__init__()
        self.bert_size = bert_size
        self.net = nn.RNN(bert_size, bert_size)
        self.lin = nn.Linear(self.bert_size*2, self.bert_size)

    def forward(self, UIs, descr, trace_screen_lengths):
        """
        UIs: batch_size x screen_size x trace_length x bert_size
        descr: batch_size x trace_length x bert_size
        """
        batch_size = UIs.size()[0]
        screen_embeddings = torch.empty(batch_size, UIs.size()[2], self.bert_size)
        # fill in total UI embeddings, trace by trace
        for batch_num in range(batch_size):
            UI_set = UIs[batch_num]
            input = torch.nn.utils.rnn.pack_padded_sequence(UI_set, trace_screen_lengths[batch_num], enforce_sorted=False)
            full_output, h = self.net(input)
            h = h[0]
            # if this isn't working, come back here and unpack output, then take last of each
            # add on screen description embedding
            concat_emb = torch.cat((h, descr[batch_num]), dim=1)
            # bring down to bert size through a linear layer
            final_emb = self.lin(concat_emb)
            screen_embeddings[batch_num] = final_emb
        # [screen_embeddings] = batch_size x trace_length x bert_size
        return screen_embeddings
        
        