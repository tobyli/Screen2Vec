
import torch.nn as nn

class UIEmbedder(nn.Module):
    def __init__(self, bert_size, num_classes ):
        self.lin = nn.Linear(bert_size + num_classes, bert_size)
