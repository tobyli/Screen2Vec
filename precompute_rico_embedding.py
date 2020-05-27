# importing libraries
from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import math
import os
import scipy
import json
from rico_dao import load_rico_screen, read_embedding_from_file, write_embedding_to_file
from rico_utils import get_all_texts_from_rico_screen, ScreenInfo
from nltk.tokenize import word_tokenize
from embedding_utils import get_embedding_from_text, get_an_aggregated_embedding_from_texts

logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO,
                            handlers=[LoggingHandler()])
sentence_bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

rico_dir = './datasets/filtered_traces'
embedding_dir = './embeddings/filtered_traces'

# populate screen_info_text_label_list_dict
rico_screen_id_text_label_list_dict = {}
rico_screen_id_screen_info_dict = {}

# load screen_info_text_label_list_dict and rico_screen_id_screen_info_dict
with open(embedding_dir + '/' + 'rico_screen_id_screen_info_dict' + '.json', 'r') as f:
    rico_screen_id_screen_info_dict_json = f.read()
    rico_screen_id_screen_info_dict = json.loads(rico_screen_id_screen_info_dict_json)

with open(embedding_dir + '/' + 'rico_screen_id_text_label_list_dict' + '.json', 'r') as f:
    rico_screen_id_text_label_list_dict_json = f.read()
    rico_screen_id_text_label_list_dict = json.loads(rico_screen_id_text_label_list_dict_json)


"""
# used for sample_rico and combined dataset
for rico_id in range(1, 1001):
    try:
        rico_screen = load_rico_screen(rico_dir, rico_id)
        package_name = rico_screen.activity_name.split('/')[0]
        screen_info = ScreenInfo(rico_id, package_name, rico_screen.activity_name)
        screen_info_text_label_list_dict[screen_info] = get_all_texts_from_rico_screen(rico_screen)
        rico_screen_id_screen_info_dict[rico_id] = screen_info
    except FileNotFoundError as e:
        print(e)
"""

# load the GloVe (https://nlp.stanford.edu/projects/glove/) (https://nlp.stanford.edu/pubs/glove.pdf) model
glove_token_embedding_dict = {}
vocab_count = 0
print ('Loading the GloVe word embeddings')
with open("./glove/glove.6B/glove.6B.300d.txt", 'r') as f:
    for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove_token_embedding_dict[word] = vector
            vocab_count += 1

print ('Finished loading the GloVe word embeddings for %d words' % vocab_count)


# load the screen info
print("Loaded %d screens" % len(rico_screen_id_text_label_list_dict))



sentence_bert_query_set = set()
for rico_screen_id, text_labels in rico_screen_id_text_label_list_dict.items():
    sentence_bert_query_set.update(text_labels)
    for text_label in text_labels:
        sentence_bert_query_set.update(word_tokenize(text_label))
    concated_label = ' '.join(text_labels)
    sentence_bert_query_set.add(concated_label)
sentence_bert_query_list = list(sentence_bert_query_set)
embeddings = sentence_bert_model.encode(sentence_bert_query_list)
text_embedding_dict = dict(zip(sentence_bert_query_list, embeddings))


# Method 1: populate method1_screen_info_embedding_dict using Sentence-BERT
for rico_screen_id, text_labels in rico_screen_id_text_label_list_dict.items():
    embedding = get_an_aggregated_embedding_from_texts(text_labels, text_embedding_dict, sentence_bert_model)
    if (not embedding is None):
        write_embedding_to_file(embedding_dir + '/method1/', rico_screen_id, embedding)


# Method 2a: populate method2_screen_info_embedding_dict using Sentence-BERT
for rico_screen_id, text_labels in rico_screen_id_text_label_list_dict.items():
    tokens = []
    # tokenize
    for text_label in text_labels:
        tokens.extend(word_tokenize(text_label))
    embedding = get_an_aggregated_embedding_from_texts(tokens, text_embedding_dict, sentence_bert_model)
    if (not embedding is None):
        write_embedding_to_file(embedding_dir + '/method2a/', rico_screen_id, embedding)

# Method 2b: populate method2_screen_info_embedding_dict using GloVe
for rico_screen_id, text_labels in rico_screen_id_text_label_list_dict.items():
    tokens = []
    # tokenize
    for text_label in text_labels:
        tokens.extend(word_tokenize(text_label))
    embedding = 0
    count = 0
    for token in tokens:
        if token.lower() in glove_token_embedding_dict:
            if (not isinstance(embedding, np.ndarray)):
                embedding = glove_token_embedding_dict[token.lower()]
                count += 1
            else:
                embedding = np.add(embedding, glove_token_embedding_dict[token.lower()])
                count += 1
    if (count > 0):
        embedding = np.divide(embedding, count)
        if (not (embedding is None or (isinstance(embedding, int)))):
            write_embedding_to_file(embedding_dir + '/method2b/', rico_screen_id, embedding)


# Method 3: populate method3_screen_info_embedding_dict using Sentence-BERT
for rico_screen_id, text_labels in rico_screen_id_text_label_list_dict.items():
    if len(text_labels) == 0:
        continue
    concated_label = ' '.join(text_labels)
    embedding = get_embedding_from_text(concated_label, text_embedding_dict, sentence_bert_model)
    if (not embedding is None):
        write_embedding_to_file(embedding_dir + '/method3/', rico_screen_id, embedding)
