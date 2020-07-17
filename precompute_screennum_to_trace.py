import json
import numpy as np
import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True, type=str, help="dataset to precompute embeddings for")
parser.add_argument("-p", "--prefix", required=True, type=str, help="prefix to output files")

args = parser.parse_args()

with open(args.prefix + 'trace_idx.json') as f:
    trace_idx = json.load(f, encoding='utf-8')

with open("/Users/lindsay/Screen2Vec/ui_layout_vectors/ui_names.json") as f:
    ui_name_list = json.load(f, encoding='utf-8')
ui_name_list = ui_name_list["ui_names"]

layout_emb_idx = []

screen_names = []

for package_dir in os.listdir(args.dataset):
    if os.path.isdir(args.dataset + '/' + package_dir):
        for trace_dir in os.listdir(args.dataset + '/' + package_dir):
            if os.path.isdir(args.dataset + '/' + package_dir + '/' + trace_dir) and (not trace_dir.startswith('.')):
                trace_id = package_dir + trace_dir[-1]
                print(trace_id)
                layout_emb_idx_trace = []
                screen_names_trace = []
                for view_hierarchy_json in os.listdir(args.dataset + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies'):
                    if view_hierarchy_json.endswith('.json') and (not view_hierarchy_json.startswith('.')):
                        json_file_path = args.dataset + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies' + '/' + view_hierarchy_json
                        try:
                            with open(json_file_path) as f:
                                screen_num = view_hierarchy_json.split('.')[0]
                                name = screen_num + ".png"
                                layout_emb_idx_trace.append(ui_name_list.index(name))
                                screen_names_trace.append(name)
                        except Exception as e:
                            print("image not found in list")
                            print(name)
                            print(e)
                            layout_emb_idx_trace.append(-1)
                layout_emb_idx.append(layout_emb_idx_trace)
                screen_names.append(screen_names_trace)

with open(args.prefix + 'layout_emb_idx.json', 'w', encoding='utf-8') as f:
    json.dump(layout_emb_idx, f, indent=4)

with open(args.prefix + 'screen_names.json', 'w', encoding='utf-8') as f:
    json.dump(screen_names, f, indent=4)
