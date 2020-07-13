from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch


class UIEmbedder(nn.Module):
    def __init__(self, bert, bert_size=768, num_classes=24, class_emb_size=4):
        super().__init__()
        self.lin = nn.Linear(bert_size + num_classes, bert_size)
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