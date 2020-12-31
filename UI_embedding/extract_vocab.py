from dataset.rico_utils import get_all_texts_from_rico_screen
from dataset.rico_dao import load_rico_screen_dict
import argparse
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# finds the vocab (all GUI labels within data) to use for comparison in training

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, type=str, help="path to rico dataset filtered traces")
parser.add_argument("-o", "--output", required=False, default="", type=str, help="path to location of output file")

args = parser.parse_args()

vocab = set()
rico_dir = args.dataset

for package_dir in os.listdir(rico_dir):
    if os.path.isdir(rico_dir + '/' + package_dir):
        # for each package directory
        for trace_dir in os.listdir(rico_dir + '/' + package_dir):
            # for each trace directory
            if os.path.isdir(rico_dir + '/' + package_dir + '/' + trace_dir) and (not trace_dir.startswith('.')):
                if os.path.isdir(rico_dir + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies'):
                    for view_hierarchy_json in os.listdir(rico_dir + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies'):
                        if view_hierarchy_json.endswith('.json') and (not view_hierarchy_json.startswith('.')):
                            json_file_path = rico_dir + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies' + '/' + view_hierarchy_json
                            with open(json_file_path) as f:
                                try:
                                    rico_screen = load_rico_screen_dict(json.load(f))
                                    text_labels = get_all_texts_from_rico_screen(rico_screen)
                                    for text in text_labels:
                                        vocab.add(text)
                                except TypeError as e:
                                    print(str(e) + ': ' + json_file_path)
vocab_list = list(vocab)

if args.output:
    output_path = args.output + 'vocab.json'
else:
    output_path = 'vocab.json'

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(vocab_list, f, indent=4)

bert = SentenceTransformer('bert-base-nli-mean-tokens')

vocab_emb = bert.encode(vocab_list)

if args.output:
    emb_output_path = args.output + "vocab_emb"
else:
    emb_output_path = "vocab_emb"

np.save(emb_output_path, vocab_emb)

