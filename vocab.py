import torch
import numpy as np
import random
from dataset.dataset import RicoDataset
from torch.utils.data import Dataset

class ScreenVocab(Dataset):
    """
    holds the collection of screens from a RicoDataset
    used for negative sampling across traces
    """
    def __init__(self, dataset:RicoDataset):
        self.dataset = dataset
        self.screens = dataset.traces
        indices, reverse_indices = self.load_indices()
        # maps overall index to trace and screen indices 
        self.indices = indices
        # maps trace and screen indices to overall index
        self.reverse_indices = reverse_indices
        self.setting = self.dataset.setting

    def get_set_of(self, num_negatives, disallowed):
        random_indices = random.sample(range(0, len(self.indices)), num_negatives)
        for dis in disallowed:
            if dis in random_indices:
                random_indices.remove(dis)
        while len(random_indices) < num_negatives:
            to_add = random.randint(0, len(self.screens))
            if (to_add not in random_indices) and (to_add not in disallowed):
                random_indices.append(to_add)
        return torch.tensor(random_indices)

    def get_negative_sample(self, num_negatives, disallowed):
        sample_indices = self.get_set_of(num_negatives, disallowed)
        idx = [self.indices[sample_idx] for sample_idx in sample_indices]
        sample_screens = [self.screens[i[0]].trace_screens[i[1]] for i in idx]
        return sample_screens

    def trace_screen_to_index(self, trace_index, screen_index):
        return self.indices[trace_index][screen_index]

    def load_indices(self):
        indices = []
        reverse_indices = []
        j = 0
        for trace_idx in range(len(self.dataset.traces)):
            trace_indices = []
            for screen_idx in range(len(self.screens[trace_idx].trace_screens)):
                indices.append((trace_idx, screen_idx))
                trace_indices.append(j)
                j+=1
            reverse_indices.append(trace_indices)
        return indices, reverse_indices
    
    def get_vocab_size(self):
        return len(self.indices)

    def negative_sample(self, num_negatives, disallowed):
        disallowed = [self.reverse_indices[dis[0]][dis[1]] for dis in disallowed]
        screens = self.get_negative_sample(num_negatives,disallowed)
        if self.setting in [0,2,6,8]:
            UIs = [torch.tensor(screen.UI_embeddings) for screen in screens]
        else:
            UIs = [torch.cat((torch.tensor(screen.UI_embeddings),torch.FloatTensor(screen.coords)), dim=1) for screen in screens]
        UI_lengths = [len(screen) for screen in UIs]
        UIs = torch.nn.utils.rnn.pad_sequence(UIs).squeeze(2).unsqueeze(0)
        descr = torch.tensor([screen.descr_emb for screen in screens]).squeeze(1).unsqueeze(0)
        if self.setting not in [0,1,6,7]:
            layouts = torch.FloatTensor([screen.layout for screen in screens]).unsqueeze(0)
        else: 
            layouts = None
        return UIs, descr, torch.tensor(UI_lengths).unsqueeze(0), layouts

    def get_all_screens(self, start_index, size):
        all_screens = []
        for trace in self.screens:
            for screen in trace.trace_screens:
                all_screens.append(screen)
        end_index = min(start_index+size, len(all_screens))
        return_screens = all_screens[start_index: end_index]
        if end_index == len(all_screens):
            end_index = -1
        if self.setting in [0,2,6,8]:
            UIs = [torch.tensor(screen.UI_embeddings) for screen in return_screens]
        else:
            UIs = [torch.cat((torch.tensor(screen.UI_embeddings),torch.FloatTensor(screen.coords)), dim=1) for screen in return_screens]
        UI_lengths = [len(screen) for screen in UIs]
        UIs = torch.nn.utils.rnn.pad_sequence(UIs).squeeze(2).unsqueeze(0)
        descr = torch.tensor([screen.descr_emb for screen in return_screens]).squeeze(1).unsqueeze(0)
        if self.setting not in [0,1,6,7]:
            layouts = torch.FloatTensor([screen.layout for screen in return_screens]).unsqueeze(0)
        else: 
            layouts = None
        return UIs, descr, torch.tensor(UI_lengths).unsqueeze(0),layouts, self.indices, self.reverse_indices, end_index
    
    def get_name(self, overall_index):
        trace_index, screen_index = self.indices[overall_index]
        screen_name = self.screens[trace_index].trace_screens[screen_index].name
        return screen_name
