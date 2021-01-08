import torch
import torch.nn as nn

# Contains classes that define our screen embeddings

class Screen2Vec(nn.Module):

    """
    Model intended to semantically embed the content of UI screens into vectors
    """

    def __init__(self, bert_size=768, additional_ui_size =0, additional_size_screen=0, net_version=5):
        """
        -bert_size (int) is the length of a Sentence-BERT text embedding
        -additional_ui_size and additional_screen_size (int) describe additional lengths for
        input depending on the network version (whether layout information is part of embedding)
        """
        super().__init__()
        self.bert_size = bert_size
        self.version = net_version
        if self.version in [0,1,2,3]:
            self.desc_size = bert_size
        else:
            self.desc_size = 0
        self.net = nn.RNN(bert_size+additional_ui_size, bert_size)
        self.lin = nn.Linear(self.bert_size + self.desc_size + additional_size_screen, self.bert_size)

    def forward(self, UIs, descr, trace_screen_lengths, layouts=None, prediction=True):
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
            if trace_screen_lengths is not None:
                input = torch.nn.utils.rnn.pack_padded_sequence(UI_set, trace_screen_lengths[batch_num], enforce_sorted=False)
            else: input = UI_set
            full_output, h = self.net(input)
            h = h[0]

            # combine UIs with rest of information
            descriptions = descr[batch_num]
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
        if not prediction and self.version ==5:
            screen_embeddings = torch.cat((screen_embeddings, descr.cpu()), dim=-1)
        return screen_embeddings



        
        