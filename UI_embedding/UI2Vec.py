from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn


class UIEmbedder(nn.Module):
    def __init__(self, bert, bert_size=768, num_classes=26, class_emb_size=6):
        super().__init__()
        self.text_embedder = bert
        self.UI_embedder = nn.Embedding(num_classes, class_emb_size)
        self.bert_size = bert_size
        self.class_size = class_emb_size

    def forward(self, text, class_name):
        text_emb = torch.as_tensor(self.text_embedder.encode(text))
        class_emb = self.UI_embedder(class_name)
        x = torch.cat((text_emb, class_emb), 1)
        for index in range(len(text)):
            if text[index] == '':
                x[index] = torch.zeros(self.bert_size + self.class_size)
        return x

class UI2Vec(nn.Module):

    """
    Model intended to semantically embed the content of a UI element into a vector
    """

    def __init__(self, bert, bert_size=768, num_classes=26, class_emb_size=6):
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

class HiddenLabelPredictorModel(nn.Module):
    """
    combines the n closest UI elements (text plus class) to predict the embedding
    of a different one on the same screen
    """
    def __init__(self, bert, bert_size, n):
        super().__init__()
        self.lin = nn.Linear(bert_size*n, bert_size)
        self.model = UI2Vec(bert)
        self.n = n
        self.bert_size = bert_size

    def forward(self, context):
        # add all of the embedded texts into a megatensor
        # if missing (less than n)- add padding
        text_embedding = self.model(context[0])
        for index in range(1, self.n):
            to_add = self.model(context[index])
            text_embedding = torch.cat((text_embedding, to_add),1)
        return self.lin(text_embedding)