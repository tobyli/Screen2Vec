import torch
class BertScreenVocab(object):
    def __init__(self, vocab_list, vocab_size, bert_model, bert_size=768):
        self.vocab_list = vocab_list
        self.bert = bert_model
        self.embeddings = self.load_embeddings()
        self.text_to_index = {}
        self.load_indices()
        self.bert_size = bert_size
    
    def load_indices(self):
        for index in range(len(self.vocab_list)):
            self.text_to_index[self.vocab_list[index]] = index

    def load_embeddings(self):
        vocab_emb = self.bert.encode(self.vocab_list)
        return vocab_emb

    def get_index(self, text):
        vec = torch.zeros(len(text))
        for index in range(len(vec)):
            vec[index] = self.text_to_index[text[index]]
        return vec

    def get_text(self, index):
        return self.vocab_list[index]

    def get_embedding(self, index, length):
        emb = torch.tensor(self.embeddings[index])
        vec = emb.repeat(length,1)
        return vec