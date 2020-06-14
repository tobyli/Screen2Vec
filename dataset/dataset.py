from torch.utils.data import Dataset, DataLoader
import torch
import random
from RicoScreen import RicoScreen


class RICODataset(Dataset):
    '''
    has traces, which have screens
    '''
    def __init__(self, data_path):
        # loads dictionaries of screen -> info lookup
        pass

    def __getitem__(self, index):
        pass
    
    def __len__():
        pass