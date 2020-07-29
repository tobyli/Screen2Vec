import torch
import torch.nn as nn
from UI_embedding.UI2Vec import UI2Vec

class Screen2Vec(nn.Module):

    """
    Model intended to semantically embed the content of UI screens into vectors
    """

    def __init__(self, bert_size=768, additional_ui_size =0, additional_size_screen=0, shrink=False):
        """
        -bert_size (int) is the length of a Sentence-BERT text embedding
        -additional_ui_size and additional_screen_size (int) describe additional lengths for
        input depending on the network version (whether layout information is part of embedding)
        -shrink (bool) is True if we intend to transform the description embedding into a lower
        dimension, False if not
        """
        super().__init__()
        self.bert_size = bert_size
        self.shrink_desc = shrink
        self.net = nn.RNN(bert_size+additional_ui_size, bert_size)
        if self.shrink_desc:
            desc_size = int(self.bert_size/2)
            self.desc_shrinker = nn.Linear(self.bert_size, desc_size)
            self.lin = nn.Linear(self.bert_size + desc_size + additional_size_screen, self.bert_size)
        else:
            self.lin = nn.Linear(self.bert_size*2+additional_size_screen, self.bert_size)

    def forward(self, UIs, descr, trace_screen_lengths, layouts=None):
        """
        UIs: batch_size x screen_size x trace_length x bert_size + additional_ui_size
        descr: batch_size x trace_length x bert_size
        trace_screen_lengths: batch_size x trace_length w
        layouts: None if not used, or batch_size x trace_length x additonal_size_screen
        """
        batch_size = UIs.size()[0]
        screen_embeddings = torch.empty(batch_size, UIs.size()[2], self.bert_size)
        # fill in total UI embeddings, trace by trace
        for batch_num in range(batch_size):
            UI_set = UIs[batch_num]
            input = torch.nn.utils.rnn.pack_padded_sequence(UI_set, trace_screen_lengths[batch_num], enforce_sorted=False)
            full_output, h = self.net(input)
            h = h[0]
            if self.shrink_desc:
                descriptions = self.desc_shrinker(descr[batch_num])
            else:
                descriptions = descr[batch_num]
            concat_emb = torch.cat((h, descriptions), dim=1)
            if layouts is not None:
                concat_emb = torch.cat((concat_emb, layouts[batch_num]), dim=1)
            # bring down to bert size through a linear layer
            final_emb = self.lin(concat_emb)
            screen_embeddings[batch_num] = final_emb
        # [screen_embeddings] = batch_size x trace_length x bert_size
        return screen_embeddings
        
        