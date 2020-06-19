
class BertScreenVocab(object):
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.text_to_index = {}
        self.load_indices()
    
    def load_indices(self):
        for index in range(len(self.vocab_list)):
            self.text_to_index[self.vocab_list[index]] = index

    def get_index(self, text):
        return self.text_to_index[text]

    def get_text(self, index):
        return self.vocab_list[index]