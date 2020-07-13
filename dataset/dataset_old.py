from torch.utils.data import Dataset, DataLoader, IterableDataset
from .playstore_scraper import get_app_description
from .rico_utils import get_all_texts_from_rico_screen, get_all_labeled_texts_from_rico_screen, ScreenInfo
from .rico_dao import load_rico_screen_dict
from sentence_transformers import SentenceTransformer
import torch
import os
import json
import random
import math



class RicoDataset(Dataset):
    '''
    has traces, which have screens
    '''
    def __init__(self, embedder, data_path, num_preds, fully_load=True):
        self.traces = []
        self.idmap = {}
        self.location = data_path
        self.UI_embedder = embedder
        self.n = num_preds + 1
        if fully_load:
            self.load_all_traces()

    def __getitem__(self, index):
        indexed_trace = self.traces[index]
        # not added unless there are at least n screens in the trace
        if len(indexed_trace.trace_screens) >= self.n:
            starting_index = random.randint(0, len(indexed_trace.trace_screens)-self.n)
            indexed_trace = indexed_trace[starting_index:starting_index+self.n]
        return [[torch.tensor(screen.embeddings) for screen in indexed_trace.trace_screens], [torch.tensor(screen.descr_emb) for screen in indexed_trace.trace_screens], [index, starting_index + self.n - 1]]
    
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
            trace_to_add = RicoTrace(self.UI_embedder, trace_data_path, True)
            if len(trace_to_add.trace_screens) >= self.n:
                self.traces.append(trace_to_add)
                self.idmap[trace_id] = len(self.traces) - 1

class RicoTrace(IterableDataset):
    """
    A list of screens
    """
    def __init__(self, embedder, data_path, fully_load):
        self.trace_screens = []
        self.location = data_path
        self.UI_embedder = embedder
        if fully_load:
            self.load_all_screens()

    def __iter__(self):
        return iter(self.trace_screens)

    def load_all_screens(self):
        descr = None
        descr_emb = None
        for view_hierarchy_json in os.listdir(self.location + '/' + 'view_hierarchies'):
            if view_hierarchy_json.endswith('.json') and (not view_hierarchy_json.startswith('.')):
                json_file_path = self.location + '/' + 'view_hierarchies' + '/' + view_hierarchy_json
                cur_screen = RicoScreen(self.UI_embedder, json_file_path, descr, descr_emb)
                descr = cur_screen.app_description
                descr_emb = cur_screen.descr_emb
                if(len(cur_screen.labeled_text) > 1):
                    self.trace_screens.append(cur_screen)

    def get_screen(self, index):
        return self.trace_screens[index]

class ScreenDataset(Dataset):
    """
    Used for training element- (not screen-) level embeddings
    Has many Rico Screens outside of their traces
    Does not include screen descriptions 
    """
    def __init__(self, rico: RicoDataset, n):
        self.screens = []
        for trace in rico.traces:
            self.screens += trace.trace_screens
        self.n = n
    
    def __getitem__(self, index):
        num_labels = len(self.screens[index].labeled_text)

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
    def __init__(self, embedder, data_path, description, description_embedding):
        self.location = data_path
        package_name, labeled_text, description = self.get_rico_info(description)
        self.labeled_text = labeled_text
        self.UI_embedder = embedder
        self.UI_embeddings = self.process_embeddings()
        self.package_name = package_name
        self.app_description = description
        self.descr_emb = self.embed_descript(description_embedding)


    def get_rico_info(self, descr):
        try:
            with open(self.location) as f:
                rico_screen = load_rico_screen_dict(json.load(f))
                package_name = rico_screen.activity_name.split('/')[0]
                labeled_text = get_all_labeled_texts_from_rico_screen(rico_screen)
                if descr is None:
                    descr = get_app_description(package_name)
            return package_name, labeled_text, descr
        except TypeError as e:
            print(str(e) + ': ' + self.location)
            return '', [], ''
    
    def get_text_info(self, index):
        if index >=0:
            return self.labeled_text[index]
        else:
            return ['', 0, [0,0,0,0]]

    def get_closest_UI_obj(self, index, n):
        bounds_to_check = self.labeled_text[index][2]
        if len(self.labeled_text) <= n:
            close_indices = [*range(len(self.labeled_text))]
        else:
            distances = [[self.distance_between(bounds_to_check, self.labeled_text[x][2]), x] 
                            for x in range(len(self.labeled_text))]
            distances.sort()
            # closest will be the same text
            close_indices = [x[1] for x in distances[1:n+1]]
        return close_indices

    def process_embeddings(self):
        embeddings = []
        for ui in self.labeled_text:
            text = [ui[0]]
            text_class = torch.tensor(ui[1]).unsqueeze(0)
            embedding = self.UI_embedder((text, text_class))
            embeddings.append(embedding)
        return embeddings

    def embed_descript(self, descr_emb):
        if descr_emb is None:
            return self.UI_embedder.embedder.text_embedder.encode([self.app_description])
        else: return descr_emb

    def distance_between(self, bounds_a, bounds_b):
        x_distance = min(abs(bounds_a[0]-bounds_b[2]), abs(bounds_a[2] - bounds_b[0]))
        y_distance = min(abs(bounds_a[1]-bounds_b[3]), abs(bounds_a[3] - bounds_b[1]))
        return math.sqrt(x_distance**2 + y_distance**2)
