import torch.nn as nn
from .embedding import UIEmbedder

class UI2Vec(nn.Module):

    """
    Model intended to semantically embed the content of a UI element into a vector
    """

    def __init__(self, bert, bert_size=768, num_classes=24, dropout=0, class_emb_size=4):
        """
        describe params here
        """
        super().__init__()
        self.embedder = UIEmbedder(bert, bert_size, num_classes)
        self.lin = nn.Linear(bert_size + class_emb_size, bert_size)

    def forward(self, input_word_labeled):
        """
        describe params here
        """
        input_word = input_word_labeled[0]
        input_label = input_word_labeled[1]
        input_vector = self.embedder(input_word, input_label)
        output = self.lin(input_vector)
        return output