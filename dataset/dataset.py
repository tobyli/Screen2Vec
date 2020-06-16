from torch.utils.data import Dataset, DataLoader, IterableDataset
from playstore_scraper import get_app_description
from rico_utils import get_all_texts_from_rico_screen, get_all_labeled_texts_from_rico_screen, ScreenInfo
from rico_dao import load_rico_screen
import torch
import os
import random



class RICODataset(Dataset):
    '''
    has traces, which have screens
    '''
    def __init__(self, data_path, fully_load=True):
        # loads dictionaries of screen -> info lookup
        self.traces = []
        self.idmap = {}
        self.location = data_path
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
            trace_to_add = RICOTrace(trace_data_path, True)
            self.traces.append(trace_to_add)
            self.idmap[trace_id] = len(self.traces) - 1

class RICOTrace(IterableDataset):
    def __init__(self, data_path, fully_load):
        self.screens = []
        self.location = data_path
        if fully_load:
            self.load_all_screens()
        pass

    def __iter__(self):
        return iter(self.screens)

    def load_all_screens(self):
        for view_hierarchy_json in os.listdir(self.location + '/' + 'view_hierarchies'):
            if view_hierarchy_json.endswith('.json') and (not view_hierarchy_json.startswith('.')):
                json_file_path = self.location + '/' + 'view_hierarchies' + '/' + view_hierarchy_json
                cur_screen = RicoScreen(json_file_path)
                self.screens.append(cur_screen)

    def get_screen(self, index):
        return self.screens[index]

class RicoScreen():
    def __init__(self, data_path):
        self.location = data_path
        package_name, description, labeled_text = self.get_rico_info()
        self.labeled_text = labeled_text
        self.package_name = package_name
        self.app_description = description

    def get_rico_info(self):
        with open(self.location) as f:
            rico_screen = load_rico_screen_dict(json.load(f))
            package_name = rico_screen.activity_name.split('/')[0]
            labeled_text = get_all_labeled_texts_from_rico_screen(rico_screen)
            description = get_app_description(package_name)
        return package_name, description, labeled_text
