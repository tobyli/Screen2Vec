from sentence_transformers import SentenceTransformer
import torch.nn as nn


class UIEmbedder(nn.Module):
    def __init__(self, bert_size=768, num_classes=24, class_emb_size=4):
        self.lin = nn.Linear(bert_size + num_classes, bert_size)
        self.text_embedder = SentenceTransformer('bert-base-nli-mean-tokens')
        self.UI_embedder = nn.Embedding(num_classes, class_emb_size)
        

    def forward(self, text, class_name):
        text_emb = self.text_embedder(text)
        class_emb = self.UItype_embedder(class_name)
        x = text_emb + class_emb
        return x