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
        return torch.as_tensor(vocab_emb)

    def get_index(self, text):
        vec = torch.LongTensor(len(text))
        for index in range(len(vec)):
            vec[index] = self.text_to_index[text[index]]
        return vec

    def get_text(self, index):
        return self.vocab_list[index]

    def get_embedding_for_cosine(self, index, length):
        # index is an integer
        emb = self.embeddings[index]
        vec = emb.repeat(length,1)
        return vec

    def get_embeddings_for_softmax(self, index):
        # index is a tensor
        result_embeddings = self.embeddings.gather(0, index)
        return result_embeddings