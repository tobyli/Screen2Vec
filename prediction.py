import torch
import torch.nn as nn
from Screen2Vec import Screen2Vec

class TracePredictor(nn.Module):
    """
    predicts the embeddings of the next screen in a trace based on its preceding screens
    """
    def __init__(self, embedding_model: Screen2Vec):
        super().__init__()
        self.model = embedding_model
        self.bert_size = self.model.bert_size
        self.combiner = nn.LSTM(self.bert_size, self.bert_size, batch_first=True)

    def forward(self, UIs, descr, trace_screen_lengths, layouts=None, cuda=True):
        """
        UIs:    embeddings of all UI elements on each screen, padded to the same length
                batch_size x screen_size x trace_length x bert_size + additional_ui_size
        descr:  Sentence BERT embeddings of app descriptions
                batch_size x trace_length x bert_size
        trace_screen_lengths: length of UIs before zero padding was performed
                batch_size x trace_length
        layouts: (None if not used in this net version) the autoencoded layout vector for the screen
                batch_size x trace_length x additonal_size_screen
        cuda:   True if TracePredictor has been sent to GPU, False if not
        """
        # embed all of the screens using Screen2Vec
        screens = self.model(UIs, descr, trace_screen_lengths, layouts)

        # take all but last element of each trace, store as context
        # last element is the desired result/target
        if cuda:
            context = torch.narrow(screens, 1, 0, screens.size()[1]-1).cuda()
            result = torch.narrow(screens, 1, screens.size()[1]-1, 1).squeeze(1).cuda()
        else:
            context = torch.narrow(screens, 1, 0, screens.size()[1]-1)
            result = torch.narrow(screens, 1, screens.size()[1]-1, 1).squeeze(1)
        
        # run screens in trace through model to predict last one
        output, (h,c) = self.combiner(context)
        return h[0], result, context