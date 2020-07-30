import torch
import torch.nn as nn

class Screen2Vec(nn.Module):

    """
    Model intended to semantically embed the content of UI screens into vectors
    """

    def __init__(self, bert_size=768, additional_ui_size =0, additional_size_screen=0, shrink=False, desc_size=768):
        """
        -bert_size (int) is the length of a Sentence-BERT text embedding
        -additional_ui_size and additional_screen_size (int) describe additional lengths for
        input depending on the network version (whether layout information is part of embedding)
        -shrink (bool) is True if we intend to transform the description embedding into a lower
        dimension, False if not
        """
        super().__init__()
        self.bert_size = bert_size
        self.desc_size = desc_size
        self.net = nn.RNN(bert_size+additional_ui_size, bert_size)
        if self.desc_size < 768 and self.desc_size > 0:
            self.desc_shrinker = nn.Linear(self.bert_size, self.desc_size)
        self.lin = nn.Linear(self.bert_size + self.desc_size + additional_size_screen, self.bert_size)

    def forward(self, UIs, descr, trace_screen_lengths, layouts=None):
        """
        UIs:    embeddings of all UI elements on each screen, padded to the same length
                batch_size x screen_size x trace_length x bert_size + additional_ui_size
        descr:  Sentence BERT embeddings of app descriptions
                batch_size x trace_length x bert_size
        trace_screen_lengths: length of UIs before zero padding was performed
                batch_size x trace_length
        layouts: (None if not used in this net version) the autoencoded layout vector for the screen
                batch_size x trace_length x additonal_size_screen
        """
        batch_size = UIs.size()[0]
        screen_embeddings = torch.empty(batch_size, UIs.size()[2], self.bert_size)
        # fill in total UI embeddings, trace by trace
        for batch_num in range(batch_size):
            UI_set = UIs[batch_num]
            input = torch.nn.utils.rnn.pack_padded_sequence(UI_set, trace_screen_lengths[batch_num], enforce_sorted=False)
            full_output, h = self.net(input)
            h = h[0]

            # combine UIs with rest of information
            # version 4: shrinking the description before combining
            if self.desc_size < 768 and self.desc_size > 0:
                descriptions = self.desc_shrinker(descr[batch_num])
            else: descriptions = descr[batch_num]
            # if training with description, concatenate it on here
            if self.desc_size > 0:
                concat_emb = torch.cat((h, descriptions), dim=1)
            else: concat_emb = h
            # if using layouts, concatenate them on here
            if layouts is not None:
                concat_emb = torch.cat((concat_emb, layouts[batch_num]), dim=1)
            
            # bring down to bert size through a linear layer
            final_emb = self.lin(concat_emb)
            screen_embeddings[batch_num] = final_emb
        # [screen_embeddings] = batch_size x trace_length x bert_size
        return screen_embeddings


class Screen2VecUse(nn.Module):

    """
    Model intended to semantically embed the content of UI screens into vectors
    """

    def __init__(self, bert_size=768, additional_ui_size =0, additional_size_screen=0, shrink=False, desc_size=768):
        """
        -bert_size (int) is the length of a Sentence-BERT text embedding
        -additional_ui_size and additional_screen_size (int) describe additional lengths for
        input depending on the network version (whether layout information is part of embedding)
        -shrink (bool) is True if we intend to transform the description embedding into a lower
        dimension, False if not
        """
        super().__init__()
        self.bert_size = bert_size
        self.desc_size = desc_size
        self.net = nn.RNN(bert_size+additional_ui_size, bert_size)
        if self.desc_size < 768 and self.desc_size > 0:
            self.desc_shrinker = nn.Linear(self.bert_size, self.desc_size)
        self.lin = nn.Linear(self.bert_size + self.desc_size + additional_size_screen, self.bert_size)

    def forward(self, UIs, descr, trace_screen_lengths, layouts=None):
        """
        UIs:    embeddings of all UI elements on each screen, padded to the same length
                batch_size x screen_size x trace_length x bert_size + additional_ui_size
        descr:  Sentence BERT embeddings of app descriptions
                batch_size x trace_length x bert_size
        trace_screen_lengths: length of UIs before zero padding was performed
                batch_size x trace_length
        layouts: (None if not used in this net version) the autoencoded layout vector for the screen
                batch_size x trace_length x additonal_size_screen
        """
        batch_size = UIs.size()[0]
        screen_embeddings = torch.empty(batch_size, UIs.size()[2], self.bert_size)
        # fill in total UI embeddings, trace by trace
        for batch_num in range(batch_size):
            UI_set = UIs[batch_num]
            input = torch.nn.utils.rnn.pack_padded_sequence(UI_set, trace_screen_lengths[batch_num], enforce_sorted=False)
            full_output, h = self.net(input)
            h = h[0]

            # combine UIs with rest of information
            if self.desc_size < 768 and self.desc_size > 0:
                descriptions = self.desc_shrinker(descr[batch_num])
            else: descriptions = descr[batch_num]
            if self.desc_size > 0:
                concat_emb = torch.cat((h, descriptions), dim=1)
            else: concat_emb = h
            if layouts is not None:
                concat_emb = torch.cat((concat_emb, layouts[batch_num]), dim=1)
            # bring down to bert size through a linear layer
            final_emb = self.lin(concat_emb)
            screen_embeddings[batch_num] = final_emb + descriptions
        # [screen_embeddings] = batch_size x trace_length x bert_size
        return screen_embeddings
        
        