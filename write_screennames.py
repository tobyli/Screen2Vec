import json
import numpy as np
import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True, type=str, help="dataset to precompute embeddings for")
parser.add_argument("-p", "--prefix", required=True, type=str, help="prefix to output files")

args = parser.parse_args()


screen_names = []

for package_dir in os.listdir(args.dataset):
    if os.path.isdir(args.dataset + '/' + package_dir):
        for trace_dir in os.listdir(args.dataset + '/' + package_dir):
            if os.path.isdir(args.dataset + '/' + package_dir + '/' + trace_dir) and (not trace_dir.startswith('.')):
                screen_names_trace = []
                for view_hierarchy_json in os.listdir(args.dataset + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies'):
                    if view_hierarchy_json.endswith('.json') and (not view_hierarchy_json.startswith('.')):
                        json_file_path = args.dataset + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies' + '/' + view_hierarchy_json
                        screen_names_trace.append(json_file_path)
                screen_names.append(screen_names_trace)

with open(args.prefix + 'screen_names.json', 'w', encoding='utf-8') as f:
    json.dump(screen_names, f, indent=4)
