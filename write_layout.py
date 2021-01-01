import argparse
import json
import numpy as np
import os
import tqdm
import torch
import torch.nn as nn

from autoencoder import ScreenLayout, LayoutAutoEncoder
from autoencoder import ScreenVisualLayout, ImageAutoEncoder

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True, type=str, help="dataset of screens to train on")
parser.add_argument("-m", "--model", type=str, default="", help="path to model")
parser.add_argument("-v", "--vis_model", type=str, default="", help="path to visual model")
parser.add_argument("-p", "--prefix", required=True, type=str, help="prefix to output files")
args = parser.parse_args()


# Creating PT data samplers and loaders:

full_model = LayoutAutoEncoder()
full_model.load_state_dict(torch.load(args.model))
model = full_model.enc
visual_model = None
if args.vis_model:
    visual_model = ImageAutoEncoder()
    visual_model.load_state_dict(torch.load(args.vis_model))
    image_model = visual_model.encoder

layout_encodings = []
image_encodings = []
i = 0
for package_dir in os.listdir(args.dataset):
    if os.path.isdir(args.dataset + '/' + package_dir):
        # for each package directory
        for trace_dir in os.listdir(args.dataset + '/' + package_dir):
            if os.path.isdir(args.dataset + '/' + package_dir + '/' + trace_dir) and (not trace_dir.startswith('.')):
                trace_id = package_dir + trace_dir[-1]
                screens = []
                screen_images = []
                for view_hierarchy_json in os.listdir(args.dataset + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies'):
                    if view_hierarchy_json.endswith('.json') and (not view_hierarchy_json.startswith('.')):
                        json_file_path = args.dataset + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies' + '/' + view_hierarchy_json
                        try:
                            screen_to_add = ScreenLayout(json_file_path)
                            screen_pixels = screen_to_add.pixels.flatten()
                            screens.append(screen_pixels)
                        except TypeError as e:
                            print(str(e) + ': ' + args.dataset)
                            screens.append(np.zeros(11200))
                        if args.vis_model:
                            try: 
                                image_path = json_file_path = args.dataset + '/' + package_dir + '/' + trace_dir + '/' + 'screenshots' + '/' + view_hierarchy_json[:-5] + ".jpg"
                                image_to_add = ScreenVisualLayout(image_path).pixels.flatten()
                                screen_images.append(image_encoding)
                            except Exception as e:
                                print(str(e))
                                screen_images.append(np.zeros(10800))
                encoded_screens = model(torch.tensor(screens).type(torch.FloatTensor)).tolist()
                layout_encodings.append(encoded_screens)
                if args.vis_model:
                    encoded_images = image_model(torch.tensor(screen_images).type(torch.FloatTensor)).tolist()
                    image_encodings.append(encoded_images)
                
                

with open(args.prefix + 'layout_embeddings.json', 'w', encoding='utf-8') as f:
    json.dump(layout_encodings, f, indent=4)

if args.vis_model:
    with open(args.prefix + 'image_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(image_encodings, f, indent=4)
    



