
import torch.nn as nn
from UI_embedding.UI2Vec import UI2Vec

class Screen2VecEmbedding(nn.Module):

    def __init__(self, model, bert, bert_size):
        super().__init__()
        self.text_embedder = UI2Vec(bert)
        self.bert = bert
        self.bert_size = bert_size
        self.text_combiner = nn.rnn(bert_size, bert_size)
    
    def forward(self, screen_info):
        batch_size = screen_info.size()[0]
        h= torch.zeros(bert_size, batch_size)
        sequence = screen_info[0]
        description = screen_info[1]
        for text in sequence:
            emb_text = self.text_embedder(text)
            output, = self.text_combiner(text,h)
        



