import json
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True, type=str, help="dataset to precompute embeddings for")
parser.add_argument("-p", "--prefix", required=True, type=str, help="prefix to output files")

args = parser.parse_args()

trace_names = []

for package_dir in os.listdir(args.dataset):
    if os.path.isdir(args.dataset + '/' + package_dir):
        # for each package directory
        package_name = package_dir
        for trace_dir in os.listdir(args.dataset + '/' + package_dir):
            if os.path.isdir(args.dataset + '/' + package_dir + '/' + trace_dir) and (not trace_dir.startswith('.')):
                trace_id = package_dir + trace_dir[-1]
                trace_names.append(trace_id)

with open(args.prefix + 'trace_names.json', 'w', encoding='utf-8') as f:
    json.dump(trace_names, f, indent=4)

