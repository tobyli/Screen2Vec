# importing libraries
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import math
import os
import scipy
import json
from rico_utils import load_rico_screen, get_all_texts_from_rico_screen, ScreenInfo, read_embedding_from_file, get_rico_id_text_label_list_dict, get_rico_id_screen_info_dict
from IPython.display import Image

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
sentence_bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# populate screen_info_text_label_list_dict

rico_dir = './datasets/sample_rico'
embedding_dir = './embeddings/sample_rico'

rico_id_screen_info_dict = get_rico_id_screen_info_dict(embedding_dir)
rico_id_text_label_list_dict = get_rico_id_text_label_list_dict(embedding_dir)

# load the screen info
print("Loaded %d screens" % len(rico_id_screen_info_dict))

method1_rico_id_embedding_dict = {}
method2a_rico_id_embedding_dict = {}
method2b_rico_id_embedding_dict = {}
method3_rico_id_embedding_dict = {}

executor = ThreadPoolExecutor(10)

def load_dict_task(embedding_dir, method_name, rico_id, dict_to_use):
    embedding = read_embedding_from_file(embedding_dir + '/' + method_name + '/', rico_id)
    if not embedding is None:
        dict_to_use[rico_id] = embedding
    return 'done'

# Method 1: populate method1_screen_info_embedding_dict using Sentence-BERT
for rico_id, screen_info in rico_id_screen_info_dict.items():
    future = executor.submit(load_dict_task, embedding_dir, 'method1', rico_id, method1_rico_id_embedding_dict)

# Method 2a: populate method2_screen_info_embedding_dict using Sentence-BERT
for rico_id, screen_info in rico_id_screen_info_dict.items():
    executor.submit(load_dict_task, embedding_dir, 'method2a', rico_id, method2a_rico_id_embedding_dict)

# Method 2b: populate method2_screen_info_embedding_dict using GloVe
for rico_id, screen_info in rico_id_screen_info_dict.items():
    executor.submit(load_dict_task, embedding_dir, 'method2b', rico_id, method2b_rico_id_embedding_dict)

# Method 3: populate method3_screen_info_embedding_dict using Sentence-BERT
for rico_id, screen_info in rico_id_screen_info_dict.items():
    executor.submit(load_dict_task, embedding_dir, 'method3', rico_id, method3_rico_id_embedding_dict)

executor.shutdown(wait=True)

# get the most similar n embeddings for src_embedding in screen_info_embedding_dict
def get_most_relevant_embeddings(src_embedding: np.ndarray, rico_id_embedding_dict: dict, n: int):
    screen_info_similarity_list = []
    for rico_id, embedding in rico_id_embedding_dict.items():
        if (embedding is None or src_embedding is None):
            continue
        if ((isinstance(embedding, int)) and embedding == 0):
            continue
        if ((isinstance(src_embedding, int)) and src_embedding == 0):
            continue
        entry = {}
        entry['rico_id'] = rico_id
        entry['score'] = scipy.spatial.distance.cosine(src_embedding, embedding)
        screen_info_similarity_list.append(entry)
    screen_info_similarity_list.sort(key=lambda x: x['score'])
    return screen_info_similarity_list[:n]


def get_most_relevant_embeddings_by_id(rico_id: int, rico_id_embedding_dict: dict, n: int):
    return get_most_relevant_embeddings(rico_id_embedding_dict[str(rico_id)], rico_id_embedding_dict, n)


def show_rico_image(rico_id: int):
    path = rico_dir + '/' + str(rico_id) + '.jpg'
    display(Image(filename=path, height=480, width=270))
    print (path)

def display_rico_result(rico_id: int):
    screen_info = rico_id_screen_info_dict[rico_id]
    show_rico_image(rico_id)
    print (screen_info.package_name)
    print (screen_info.activity_name)
    print (rico_id_text_label_list_dict[rico_id])


source_id = 376

print("source")
show_rico_image(source_id)
print('\n')

print("method1")
for entry in get_most_relevant_embeddings_by_id(source_id, method1_rico_id_embedding_dict, 10):
    display_rico_result(entry['rico_id'])
    print("Similarity Score: ", entry['score'])
print('\n')

print("method2a")
for entry in get_most_relevant_embeddings_by_id(source_id, method2a_rico_id_embedding_dict, 10):
    display_rico_result(entry['rico_id'])
    print("Similarity Score: ", entry['score'])
print('\n')

print("method2b")
for entry in get_most_relevant_embeddings_by_id(source_id, method2b_rico_id_embedding_dict, 10):
    display_rico_result(entry['rico_id'])
    print("Similarity Score: ", entry['score'])
print('\n')

print("method3")
for entry in get_most_relevant_embeddings_by_id(source_id, method3_rico_id_embedding_dict, 10):
    display_rico_result(entry['rico_id'])
    print("Similarity Score: ", entry['score'])
print('\n')