from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import torch
import os
import json
import random
import math
import numpy as np

class TesterRicoDataset(Dataset):
    '''
    has traces, which have screens
    '''
    def __init__(self, num_preds, ui, ui_e, d, d_e, l, net_version=0, fully_load=True, screen_names=None):
        self.traces = []
        self.n = num_preds + 1
        self.ui_e = ui_e
        self.d_e = d_e
        self.setting = net_version
        self.s_n = screen_names
        if fully_load:
            self.load_all_traces(ui, d, l)

    def __getitem__(self, index):
        indexed_trace = self.traces[index]
        # not added unless there are at least n screens in the trace
        traces = []
        if len(indexed_trace.trace_screens) >= self.n:
            starting_indices = range(0, len(indexed_trace.trace_screens)-self.n +1)
            for st_idx in starting_indices:
                traces.append(self.get_item_part(index, st_idx))
        return traces

    def get_item_part(self, trace_index, starting_screen_index):
        indexed_trace = self.traces[trace_index]
        # not added unless there are at least n screens in the trace
        screens = indexed_trace.trace_screens[starting_screen_index:starting_screen_index+self.n-1]
        if self.setting==0 or self.setting==6:
            return [[torch.tensor(screen.UI_embeddings) for screen in screens], [screen.descr_emb for screen in screens], [trace_index, starting_screen_index + self.n - 1], None]
        elif self.setting==1 or self.setting==7:
            return [[torch.cat((torch.tensor(screen.UI_embeddings),torch.FloatTensor(screen.coords)), dim=1) for screen in screens], [screen.descr_emb for screen in screens], [trace_index, starting_screen_index + self.n - 1], None]
        elif self.setting==2 or self.setting==8:
            return [[torch.tensor(screen.UI_embeddings) for screen in screens], [screen.descr_emb for screen in screens], [trace_index, starting_screen_index + self.n - 1], [screen.layout for screen in screens]]
        else:
            return [[torch.cat((torch.tensor(screen.UI_embeddings),torch.FloatTensor(screen.coords)), dim=1) for screen in screens], [screen.descr_emb for screen in screens], [trace_index, starting_screen_index + self.n - 1], [screen.layout for screen in screens]]

    def __len__(self):
        return len(self.traces)

    def load_all_traces(self, ui, d, l):
        if self.setting in [0,1,6,7]:
            for trace_idx in range(len(self.d_e)):
                if self.s_n:
                    self.load_trace(ui[trace_idx], self.ui_e[trace_idx], d[trace_idx], self.d_e[trace_idx], None, self.s_n[trace_idx])
                else:
                    self.load_trace(ui[trace_idx], self.ui_e[trace_idx], d[trace_idx], self.d_e[trace_idx])
        else:
            for trace_idx in range(len(self.d_e)):
                if self.s_n:
                    self.load_trace(ui[trace_idx], self.ui_e[trace_idx], d[trace_idx], self.d_e[trace_idx], l[trace_idx],self.s_n[trace_idx])
                else:
                    self.load_trace(ui[trace_idx], self.ui_e[trace_idx], d[trace_idx], self.d_e[trace_idx], l[trace_idx])

    def load_trace(self, ui, ui_e, d, d_e, l=None, s_n=None):
        # loads a trace
        trace_to_add = RicoTrace(ui, ui_e, d, d_e, l, self.setting, s_n)
        if len(trace_to_add.trace_screens) >= self.n and d!="":
            self.traces.append(trace_to_add)

class PrecompRicoDataset(Dataset):
    '''
    has traces, which have screens
    '''
    def __init__(self, num_preds, ui, ui_e, d, d_e, l, net_version=0, fully_load=True, screen_names=None):
        self.traces = []
        self.n = num_preds + 1
        self.ui_e = ui_e
        self.d_e = d_e
        self.setting = net_version
        self.s_n = screen_names
        if fully_load:
            self.load_all_traces(ui, d, l)

    def __getitem__(self, index):
        screens = self.traces[index].trace_screens
        if self.setting==0 or self.setting==6:
            return [[torch.tensor(screen.UI_embeddings) for screen in screens], [screen.descr_emb for screen in screens], [index, starting_index + self.n - 1], None]
        elif self.setting==1 or self.setting==7:
            return [[torch.cat((torch.tensor(screen.UI_embeddings),torch.FloatTensor(screen.coords)), dim=1) for screen in screens], [screen.descr_emb for screen in screens], [index, starting_index + self.n - 1], None]
        elif self.setting==2  or self.setting==8:
            return [[torch.tensor(screen.UI_embeddings) for screen in screens], [screen.descr_emb for screen in screens], [index, starting_index + self.n - 1], [screen.layout for screen in screens]]
        else:
            return [[torch.cat((torch.tensor(screen.UI_embeddings),torch.FloatTensor(screen.coords)), dim=1) for screen in screens], [screen.descr_emb for screen in screens], [index, starting_index + self.n - 1], [screen.layout for screen in screens]]
    
    def __len__(self):
        return len(self.traces)

    def load_all_traces(self, ui, d, l):
        if self.setting in [0,1,6,7]:
            for trace_idx in range(len(self.d_e)):
                if self.s_n:
                    self.load_trace(ui[trace_idx], self.ui_e[trace_idx], d[trace_idx], self.d_e[trace_idx], None, self.s_n[trace_idx])
                else:
                    self.load_trace(ui[trace_idx], self.ui_e[trace_idx], d[trace_idx], self.d_e[trace_idx])
        else:
            for trace_idx in range(len(self.d_e)):
                if self.s_n:
                    self.load_trace(ui[trace_idx], self.ui_e[trace_idx], d[trace_idx], self.d_e[trace_idx], l[trace_idx],self.s_n[trace_idx])
                else:
                    self.load_trace(ui[trace_idx], self.ui_e[trace_idx], d[trace_idx], self.d_e[trace_idx], l[trace_idx])

    def load_trace(self, ui, ui_e, d, d_e, l=None, s_n=None):
        # loads a trace
        trace_to_add = RicoTrace(ui, ui_e, d, d_e, l, self.setting, s_n)
        self.traces.append(trace_to_add)

class RicoDataset(Dataset):
    '''
    has traces, which have screens
    '''
    def __init__(self, num_preds, ui, ui_e, d, d_e, l, net_version=0, fully_load=True, screen_names=None):
        self.traces = []
        self.n = num_preds + 1
        self.ui_e = ui_e
        self.d_e = d_e
        self.setting = net_version
        self.s_n = screen_names
        if fully_load:
            self.load_all_traces(ui, d, l)

    def __getitem__(self, index):
        indexed_trace = self.traces[index]
        # not added unless there are at least n screens in the trace
        if len(indexed_trace.trace_screens) >= self.n:
            starting_index = random.randint(0, len(indexed_trace.trace_screens)-self.n)
            screens = indexed_trace.trace_screens[starting_index:starting_index+self.n-1]
        if self.setting==0 or self.setting==6:
            return [[torch.tensor(screen.UI_embeddings) for screen in screens], [screen.descr_emb for screen in screens], [index, starting_index + self.n - 1], None]
        elif self.setting==1 or self.setting==7:
            return [[torch.cat((torch.tensor(screen.UI_embeddings),torch.FloatTensor(screen.coords)), dim=1) for screen in screens], [screen.descr_emb for screen in screens], [index, starting_index + self.n - 1], None]
        elif self.setting==2 or self.setting==8:
            return [[torch.tensor(screen.UI_embeddings) for screen in screens], [screen.descr_emb for screen in screens], [index, starting_index + self.n - 1], [screen.layout for screen in screens]]
        else:
            return [[torch.cat((torch.tensor(screen.UI_embeddings),torch.FloatTensor(screen.coords)), dim=1) for screen in screens], [screen.descr_emb for screen in screens], [index, starting_index + self.n - 1], [screen.layout for screen in screens]]

    def __len__(self):
        return len(self.traces)

    def load_all_traces(self, ui, d, l):
        if self.setting in [0,1,6,7]:
            for trace_idx in range(len(self.d_e)):
                if self.s_n:
                    self.load_trace(ui[trace_idx], self.ui_e[trace_idx], d[trace_idx], self.d_e[trace_idx], None, self.s_n[trace_idx])
                else:
                    self.load_trace(ui[trace_idx], self.ui_e[trace_idx], d[trace_idx], self.d_e[trace_idx])
        else:
            for trace_idx in range(len(self.d_e)):
                if self.s_n:
                    self.load_trace(ui[trace_idx], self.ui_e[trace_idx], d[trace_idx], self.d_e[trace_idx], l[trace_idx],self.s_n[trace_idx])
                else:
                    self.load_trace(ui[trace_idx], self.ui_e[trace_idx], d[trace_idx], self.d_e[trace_idx], l[trace_idx])

    def load_trace(self, ui, ui_e, d, d_e, l=None, s_n=None):
        # loads a trace
        trace_to_add = RicoTrace(ui, ui_e, d, d_e, l, self.setting, s_n)
        if len(trace_to_add.trace_screens) >= self.n and d!="":
            self.traces.append(trace_to_add)
            

class RicoTrace():
    """
    A list of screens
    """
    def __init__(self, ui, ui_e, d, d_e, l, setting=0, s_n = None):
        self.ui_e = ui_e
        self.d_e = d_e
        self.trace_screens = []
        self.setting = setting
        self.load_all_screens(ui, d, l, s_n)
        if s_n:
            self.names = s_n[0].split('/')[-4:-2]

    def __iter__(self):
        return iter(self.trace_screens)

    def load_all_screens(self, ui, d, l, s_n):
        
        for screen_idx in range(len(self.ui_e)):
            if s_n:
                name = s_n[screen_idx]
            else: 
                name = None
            if self.setting in [0,1,6,7]:
                screen_to_add = RicoScreen(ui[screen_idx], self.ui_e[screen_idx], d, self.d_e, None, self.setting, name)
            else:
                screen_to_add = RicoScreen(ui[screen_idx], self.ui_e[screen_idx], d, self.d_e, l[screen_idx], self.setting, name)
            if len(screen_to_add.UI_embeddings) > 0:
                self.trace_screens.append(screen_to_add)

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
    def __init__(self, ui, ui_e, d, d_e, l, setting=0, s_n=None):
        self.labeled_text = ui
        self.UI_embeddings = ui_e
        self.descr = d
        self.descr_emb = d_e
        self.layout = l
        self.setting = setting
        self.name = s_n
        if setting not in [0,2,6]:
            self.coords = self.load_coords()
        else:
            self.coords = []

    
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

    def distance_between(self, bounds_a, bounds_b):
        x_distance = min(abs(bounds_a[0]-bounds_b[2]), abs(bounds_a[2] - bounds_b[0]))
        y_distance = min(abs(bounds_a[1]-bounds_b[3]), abs(bounds_a[3] - bounds_b[1]))
        return math.sqrt(x_distance**2 + y_distance**2)

    def load_coords(self):
        coords = []
        for ui in self.labeled_text:
            coords.append(ui[2])
        return coords
