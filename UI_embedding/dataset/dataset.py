from torch.utils.data import Dataset, DataLoader, IterableDataset
from .playstore_scraper import get_app_description
from .rico_utils import get_all_texts_from_rico_screen, get_all_labeled_uis_from_rico_screen, get_hierarchy_dist_from_rico_screen, ScreenInfo
from .rico_dao import load_rico_screen_dict
from sentence_transformers import SentenceTransformer
import torch
import os
import json
import random
import math
import numpy as np



class RicoDataset(Dataset):
    '''
    has traces, which have screens
    '''
    def __init__(self, data_path, fully_load=True, hierarchy=False):
        self.traces = []
        self.idmap = {}
        self.location = data_path
        self.hierarchy = hierarchy
        self.token_model = SentenceTransformer('bert-base-nli-mean-tokens')
        if fully_load:
            self.load_all_traces()

    def __getitem__(self, index):
        return self.traces[index]
    
    def __len__(self):
        return len(self.traces)

    def load_all_traces(self):
        for package_dir in os.listdir(self.location):
            if os.path.isdir(self.location + '/' + package_dir):
                # for each package directory
                for trace_dir in os.listdir(self.location + '/' + package_dir):
                    if os.path.isdir(self.location + '/' + package_dir + '/' + trace_dir) and (not trace_dir.startswith('.')):
                        trace_id = package_dir + trace_dir[-1]
                        self.load_trace(trace_id, self.location + '/' + package_dir + '/' + trace_dir)

    def load_trace(self, trace_id, trace_data_path):
        # loads a trace
        # trace_id should come from trace_data_path
        if not self.idmap.get(trace_id):
            trace_to_add = RicoTrace(trace_data_path, True, self.hierarchy)
            self.traces.append(trace_to_add)
            self.idmap[trace_id] = len(self.traces) - 1

class RicoTrace(IterableDataset):
    """
    A list of screens
    """
    def __init__(self, data_path, fully_load, hierarchy=False):
        self.trace_screens = []
        self.location = data_path
        self.hierarchy = hierarchy
        if fully_load:
            self.load_all_screens()
        pass

    def __iter__(self):
        return iter(self.trace_screens)

    def load_all_screens(self):
        for view_hierarchy_json in os.listdir(self.location + '/' + 'view_hierarchies'):
            if view_hierarchy_json.endswith('.json') and (not view_hierarchy_json.startswith('.')):
                json_file_path = self.location + '/' + 'view_hierarchies' + '/' + view_hierarchy_json
                cur_screen = RicoScreen(json_file_path, self.hierarchy)
                if(len(cur_screen.labeled_uis) > 1):
                    self.trace_screens.append(cur_screen)

    def get_screen(self, index):
        return self.trace_screens[index]

class ScreenDataset(Dataset):
    """
    Used for training element- (not screen-) level embeddings
    Has many Rico Screens outside of their traces
    Does not include screen descriptions 
    """
    def __init__(self, rico_traces, n, hierarchy=False):
        self.screens = []
        for trace in rico_traces:
            self.screens += trace.trace_screens
        self.n = n
    
    def __getitem__(self, index):
        num_labels = len(self.screens[index].labeled_uis)

        hidden_index = random.randint(0, num_labels-1)
        hidden_text = self.screens[index].get_text_info(hidden_index)[:2]
        other_indices = self.screens[index].get_closest_UI_obj(hidden_index, self.n)
        while len(other_indices) < self.n:
            other_indices.append(-1)
        other_text = [self.screens[index].get_text_info(i)[:2] for i in other_indices]

        return [hidden_text, other_text]
    
    def __len__(self):
        return len(self.screens)

class RicoScreen():
    """
    The information from one screenshot of a app- package name
    and labeled text (text, class, and location)
    """
    def __init__(self, data_path, hierarchy=False):
        self.location = data_path
        package_name, labeled_uis = self.get_rico_info()
        self.labeled_uis = labeled_uis
        self.package_name = package_name
        self.hierarchy = hierarchy
        if self.hierarchy:
            self.distances = np.empty((0,0))
            self.load_distances()
        #self.app_description = description

    def get_rico_info(self):
        try:
            with open(self.location) as f:
                rico_screen = load_rico_screen_dict(json.load(f))
                package_name = rico_screen.activity_name.split('/')[0]
                labeled_uis = get_all_labeled_uis_from_rico_screen(rico_screen)
            return package_name, labeled_uis # , description
        except TypeError as e:
            print(str(e) + ': ' + self.location)
            return '', []

    def load_distances(self):
        try:
            with open(self.location) as f:
                rico_screen = load_rico_screen_dict(json.load(f))
                self.distances = get_hierarchy_dist_from_rico_screen(rico_screen, len(self.labeled_uis))
        except TypeError as e:
            print(str(e) + ': ' + self.location)
            return np.empty((0,0))
    
    def get_text_info(self, index):
        if index >=0:
            return self.labeled_uis[index]
        else:
            return ['', 0, [0,0,0,0]]

    def get_closest_UI_obj(self, index, n):
        bounds_to_check = self.labeled_uis[index][2]
        if len(self.labeled_uis) <= n:
            close_indices = [*range(len(self.labeled_uis))]
        else:
            if not self.hierarchy:
                distances = [[self.distance_between(bounds_to_check, self.labeled_uis[x][2]), x] 
                            for x in range(len(self.labeled_uis))]
            else: 
                distances = [[self.distances[index,x], x]
                            for x in range(len(self.labeled_uis))]
            distances.sort()
            # closest will be the same text
            close_indices = [x[1] for x in distances[1:n+1]]
        return close_indices

    def distance_between(self, bounds_a, bounds_b):
        x_distance = min(abs(bounds_a[0]-bounds_b[2]), abs(bounds_a[2] - bounds_b[0]))
        y_distance = min(abs(bounds_a[1]-bounds_b[3]), abs(bounds_a[3] - bounds_b[1]))
        return math.sqrt(x_distance**2 + y_distance**2)
