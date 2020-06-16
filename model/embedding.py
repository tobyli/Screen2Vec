
import torch.nn as nn

class Screen2VecEmbedding(nn.Module):

    def __init__(self):
        self.text = TextLabelEmbedding
    
    def forward(self, sequence):
        x = self.UIEmbedding(sequence) # + bert embedding of 
        



class UIEmbedder(nn.Module):
    def __init__(self, bert_size=768, num_classes ):
        self.lin = nn.Linear(bert_size + num_classes, bert_size)
