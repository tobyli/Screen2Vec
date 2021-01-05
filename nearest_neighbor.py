import argparse
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
import scipy
import random
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import os

# contains code for running nearest neighbor experiment

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--emb_path", type=str, default="", help="path to stored embeddings")
    parser.add_argument("-s", "--selected", type=str, default="", help="path to query screen")
    parser.add_argument("-c", "--command", type=str, default="", help="natural language command to find relevant screens for")
    parser.add_argument("-n", "--n", type=int, default=5, help="number of relevant screens to find")

    args = parser.parse_args()
    
    with open(args.emb_path) as f:
        embeddings = json.load(f, encoding='utf-8')
    if args.selected:
        print(get_most_relevant_embeddings(args.selected, embeddings, args.n))
    elif args.command:
        bert = SentenceTransformer('bert-base-nli-mean-tokens')
        src_emb = bert.encode([args.command])
        print(get_most_relevant_embeddings_nl(src_emb, embeddings,5))
    else:
        n = 0
        keys = list(embeddings.keys())
        while n < 10:
            n+=1
            key = random.choice(keys)
            print("___________________________________")
            print(key)
            print("_-_-_-_-_-_-")
            print(get_most_relevant_embeddings(key, embeddings, 5))

if __name__ == "__main__":
    main()

def get_full_path_from_relative_path_if_not_available(file_path, home_dataset_path):
    if (not os.path.exists(file_path)):
        return home_dataset_path + file_path
    else:
        return file_path
    
    
def get_hierachy_for_json_path(json_path, home_dataset_path):
    with open(get_full_path_from_relative_path_if_not_available(json_path, home_dataset_path)) as f:
        data = json.load(f)
        return data

def get_most_relevant_embeddings(src_id, rico_id_embedding_dict: dict, n: int, home_dataset_path, filter_duplicated_activity = False):
    try:
        src_embedding = rico_id_embedding_dict[src_id]
    except KeyError as e:
        # this is only for testing nearest neighbors, NOT VALID
        try:
            src_id = src_id.split("/")
            front = src_id[:-4]
            back = src_id[-4:]
            front = "/".join(front)
            back = "/".join(back)
            src_id = "//".join([front,back])
            src_embedding = rico_id_embedding_dict[src_id]
        except KeyError as e:
            src_embedding = list(rico_id_embedding_dict.values())[0]
            
    screen_info_similarity_list = []
    app_name_1 = src_id.split("/")[-4]
    for rico_id, embedding in rico_id_embedding_dict.items():
        if (embedding is None or src_embedding is None):
            continue
        if ((isinstance(embedding, int)) and embedding == 0):
            continue
        if ((isinstance(src_embedding, int)) and src_embedding == 0):
            continue
        app_name_2 = rico_id.split("/")[-4]
        if app_name_1 == app_name_2:
            continue
        entry = {}
        entry['rico_id'] = rico_id
        entry['score'] = scipy.spatial.distance.cosine(src_embedding, embedding)
        if (entry['rico_id'] is not None and not np.isnan (entry['score'])):  
            screen_info_similarity_list.append(entry)
    screen_info_similarity_list.sort(key=lambda x: x['score'])
    
    if filter_duplicated_activity:
        filtered_result = []      
        activity_name_set = set()
        for entry in screen_info_similarity_list:
            json_data = get_hierachy_for_json_path(entry['rico_id'], home_dataset_path)
            activity_name = json_data['activity_name']
            if (activity_name not in activity_name_set):
                activity_name_set.add(activity_name)
                filtered_result.append(entry)
            if (len(filtered_result) >= n):
                break
        return filtered_result
    return screen_info_similarity_list[0:n if n <= len(screen_info_similarity_list) else len(screen_info_similarity_list)]

def vector_compose(screen1, screen2, screen3, emb_dict):
    """
    screen1 + screen2 - screen3
    """
    result = emb_dict[screen1] + emb_dict[screen2] - emb_dict[screen3]
    closest = float('inf')
    close_screen = ''
    for id, embedding in emb_dict.items():
        if (embedding is None):
            continue
        if ((isinstance(embedding, int)) and embedding == 0):
            continue
        dist = scipy.spatial.distance.cosine(result, embedding)
        if dist<closest:
            closest = dist
            close_screen = id
    return close_screen

def get_most_relevant_embeddings_nl(src_embedding, rico_id_embedding_dict: dict, n:int):
    screen_info_similarity_list = []
    for rico_id, embedding in rico_id_embedding_dict.items():
        if (embedding is None or src_embedding is None):
            continue
        if ((isinstance(embedding, int)) and embedding == 0):
            continue
        if ((isinstance(src_embedding, int)) and src_embedding == 0):
            continue
        if (len(embedding) > len(src_embedding)):
            shrinked_embedding = embedding[0:len(src_embedding)]
        entry = {}
        entry['rico_id'] = rico_id
        entry['score'] = scipy.spatial.distance.cosine(src_embedding, shrinked_embedding)
        screen_info_similarity_list.append(entry)
    screen_info_similarity_list.sort(key=lambda x: x['score'])
    return screen_info_similarity_list[0:n]


