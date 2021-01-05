import argparse
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
import os
import scipy
import numpy as np
from dataset.rico_utils import get_all_texts_from_rico_screen
from dataset.rico_dao import load_rico_screen_dict
from sentence_transformers import SentenceTransformer
from autoencoder import ScreenLayout, LayoutAutoEncoder, ScreenVisualLayout, ImageAutoEncoder

# generates all the embeddings for the baseline models

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True, type=str, help="dataset to precompute embeddings for")
parser.add_argument("-o", "--output", required=True, type=str, help="path to store embedding output files")
parser.add_argument("-l", "--layout_model", required=True, type=str, help="path to layout autoencoder model")
parser.add_argument("-v", "--visual_model", required=True, type=str, help="path to visual autoencoder model")

args = parser.parse_args()

bert = SentenceTransformer('bert-base-nli-mean-tokens')
bert_size = 768

word_embeddings = {}
layout_embeddings = {}
visual_embeddings = {}

layout_autoencoder = LayoutAutoEncoder()
layout_autoencoder.load_state_dict(torch.load(args.layout_model))

visual_autoencoder = ImageAutoEncoder()
visual_autoencoder.load_state_dict(torch.load(args.visual_model))

i = 0
for package_dir in os.listdir(args.dataset):
    if os.path.isdir(args.dataset + '/' + package_dir):
        for trace_dir in os.listdir(args.dataset + '/' + package_dir):
            if os.path.isdir(args.dataset + '/' + package_dir + '/' + trace_dir) and (not trace_dir.startswith('.')):
                trace_id = package_dir + trace_dir[-1]
                print(i)
                i+=1
                for view_hierarchy_json in os.listdir(args.dataset + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies'):
                    if view_hierarchy_json.endswith('.json') and (not view_hierarchy_json.startswith('.')):
                        json_file_path = package_dir + '/' + trace_dir + '/' + 'view_hierarchies' + '/' + view_hierarchy_json
                        try:
                            with open(args.dataset + '/' +  json_file_path) as f:
                                rico_screen = load_rico_screen_dict(json.load(f))
                                text = get_all_texts_from_rico_screen(rico_screen)
                                if text == []:
                                    text = [""]
                        except TypeError as e:
                            print(str(e) + ': ' + args.dataset)
                            text = [""]
                        word_embs = bert.encode(text)
                        if len(word_embs) > 0:
                            word_avg_emb = np.mean(word_embs, axis=0)
                        else:
                            word_avg_emb = word_embs[0]
                        word_embeddings[json_file_path] = word_avg_emb.tolist()

                        layout_screen = ScreenLayout(args.dataset + '/' +  json_file_path)
                        screen_pix = torch.from_numpy(layout_screen.pixels.flatten()).type(torch.FloatTensor)
                        layout_emb = layout_autoencoder.enc(screen_pix)
                        layout_embeddings[json_file_path] = layout_emb.detach().tolist()
                        
                        vis_screen = ScreenVisualLayout(args.dataset + "/" + package_dir + '/' + trace_dir + '/' + 'screenshots' + '/' + view_hierarchy_json.split(".")[0] + ".jpg")
                        screen_pix = torch.from_numpy(vis_screen.pixels.flatten()).type(torch.FloatTensor)/255
                        vis_emb = visual_autoencoder.encoder(screen_pix)
                        visual_embeddings[json_file_path] = vis_emb.detach().tolist()


with open(args.output + 'text_baseline.json', 'w', encoding='utf-8') as f:
    json.dump(word_embeddings, f, indent=4)

with open(args.output + 'layout_baseline.json', 'w', encoding='utf-8') as f:
    json.dump(layout_embeddings, f, indent=4)

with open(args.output + 'visual_baseline.json', 'w', encoding='utf-8') as f:
    json.dump(visual_embeddings, f, indent=4)



