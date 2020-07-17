import torch
import numpy as np
import random
from dataset.dataset import RicoDataset

class ScreenVocab(object):
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
        if self.setting in [0,2]:
            UIs = [torch.tensor(screen.UI_embeddings) for screen in screens]
        else:
            UIs = [torch.cat((torch.tensor(screen.UI_embeddings),torch.FloatTensor(screen.coords)), dim=1) for screen in screens]
        UI_lengths = [len(screen) for screen in UIs]
        UIs = torch.nn.utils.rnn.pad_sequence(UIs).squeeze(2).unsqueeze(0)
        if self.setting in [0,1]:
            descr = torch.tensor([screen.descr_emb for screen in screens]).squeeze(1).unsqueeze(0)
        else:
            descr = torch.tensor([np.concatenate((screen.descr_emb, screen.layout)) for screen in screens]).squeeze(1).unsqueeze(0)
        return UIs, descr, torch.tensor(UI_lengths).unsqueeze(0)

        #Note to self: may need to add tensor dimension for "batch"