import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from collections.abc import Iterable

import numpy as np
import os
import json
import tqdm
from PIL import Image


class ScreenLayout():

    def __init__(self, screen_path):
        self.pixels = np.full((100,56,2), 0, dtype=float)
        self.vert_scale = 100/2560
        self.horiz_scale = 56/1440
        self.load_screen(screen_path)

    def load_screen(self, screen_path):
        with open(screen_path) as f:
            hierarchy = json.load(f, encoding='utf-8')
        try:
            root = hierarchy["activity"]["root"]
            self.load_screen_contents(root)
        except KeyError as e:
            print(e)
        except TypeError as e:
            print(e)

    def load_screen_contents(self, node):
        results = []
        if 'children' in node and isinstance(node['children'], Iterable):
            for child_node in node['children']:
                if (isinstance(child_node, dict)):
                    self.load_screen_contents(child_node)
        else:
            try:
                if ("visible-to-user" in node and node["visible-to-user"]) or ("visible_to_user" in node and node["visible_to_user"]):
                    bounds = node["bounds"]
                    x1 = int(bounds[0]*self.horiz_scale)
                    y1 = int(bounds[1]*self.vert_scale)
                    x2 = int(bounds[2]*self.horiz_scale)
                    y2 = int(bounds[3]*self.vert_scale)
                    if 'text' in node and node['text'] and node['text'].strip():
                        #append in 'blue' ([0]) here
                        self.pixels[y1:y2,x1:x2,0] = 1
                    else: 
                        #append in 'red' ([1]) here
                        self.pixels[y1:y2,x1:x2,1] = 1
            except KeyError as e:
                print(e)
                    
    def convert_to_image(self):
        p = np.full((100,56,3), 255, dtype=np.uint)
        for y in range(len(self.pixels)):
            for x in range(len(self.pixels[0])):
                if (self.pixels[y][x] == [1,0]).all() or (self.pixels[y][x] == [1,1]).all():
                    p[y][x] = [0,0,255]
                elif (self.pixels[y][x] == [0,1]).all():
                    p[y][x] = [255,0,0]
        im = Image.fromarray(p.astype(np.uint8))
        im.save("example.png")




class ScreenLayoutDataset(Dataset):
    def __init__(self, dataset_path):
        self.screens = self.load_screens(dataset_path)

    def __len__(self):
        return len(self.screens)

    def __getitem__(self, index):
        return torch.from_numpy(self.screens[index].pixels.flatten()).type(torch.FloatTensor)

    def load_screens(self, dataset_path):
        screens = []
        for fn in os.listdir(dataset_path):
            if fn.endswith('.json'):
                screen_layout = ScreenLayout(dataset_path + '/' + fn)
                screens.append(screen_layout)
        return screens

class ScreenVisualLayout():

    def __init__(self, screen_path):
        self.pixels = self.load_screen(screen_path)

    def load_screen(self, screen_path):
        im = Image.open(screen_path, 'r')
        im = im.resize((90,160))
        return np.array(im)

class ScreenVisualLayoutDataset(Dataset):
    def __init__(self, dataset_path):
        self.screens = self.load_screens(dataset_path)

    def __len__(self):
        return len(self.screens)

    def __getitem__(self, index):
        return torch.from_numpy(self.screens[index].pixels.flatten()).type(torch.FloatTensor)

    def load_screens(self, dataset_path):
        screens = []
        for fn in os.listdir(dataset_path):
            if fn.endswith('.jpg'):
                screen_layout = ScreenVisualLayout(dataset_path + '/' + fn)
                screens.append(screen_layout)
        return screens


class LayoutEncoder(nn.Module):

    def __init__(self):
        super(LayoutEncoder, self).__init__()

        self.e1 = nn.Linear(11200, 2048)
        self.e2 = nn.Linear(2048, 256)
        self.e3 = nn.Linear(256, 64)


    def forward(self, input):
        encoded = F.relu(self.e3(F.relu(self.e2(F.relu(self.e1(input))))))
        return encoded


class LayoutDecoder(nn.Module):

    def __init__(self):
        super(LayoutDecoder, self).__init__()

        self.d1 = nn.Linear(64,256)
        self.d2 = nn.Linear(256, 2048)
        self.d3 = nn.Linear(2048, 11200)

    def forward(self, input):
        decoded = F.relu(self.d3(F.relu(self.d2(F.relu(self.d1(input))))))
        return decoded

class LayoutAutoEncoder(nn.Module):

    def __init__(self):
        super(LayoutAutoEncoder, self).__init__()

        self.enc = LayoutEncoder()
        self.dec = LayoutDecoder()

    def forward(self, input):
        return F.relu(self.dec(self.enc(input)))

class ImageLayoutEncoder(nn.Module):

    def __init__(self):
        super(ImageLayoutEncoder, self).__init__()

        self.lin = nn.Linear(43200, 11200)
        self.layout_encoder = LayoutEncoder()
        for param in self.layout_encoder.parameters():
            param.requires_grad = False

    def forward(self, input):
        return F.relu(self.layout_encoder(self.lin(input)))
    

class ImageLayoutDecoder(nn.Module):

    def __init__(self):
        super(ImageLayoutDecoder, self).__init__()

        self.lin = nn.Linear(11200, 43200)
        self.layout_decoder = LayoutDecoder()
        for param in self.layout_decoder.parameters():
            param.requires_grad = False
        

    def forward(self, input):
        return self.lin(self.layout_decoder(input))

class ImageAutoEncoder(nn.Module):

    def __init__(self):
        super(ImageAutoEncoder, self).__init__()

        self.encoder = ImageLayoutEncoder()
        self.decoder = ImageLayoutDecoder()

    def forward(self, input):
        return self.decoder(self.encoder(input))


class LayoutTrainer():
    def __init__(self, auto_enc: LayoutAutoEncoder, dataloader_train, dataloader_test, l_rate):
        self.model = auto_enc
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=l_rate)
        self.train_data = dataloader_train
        self.test_data = dataloader_test

    def train(self, epoch):
        loss = self.iteration(epoch, self.train_data)
        return loss

    def test(self, epoch):
        loss = self.iteration(epoch, self.test_data, train=False)
        return loss

    def iteration(self, epoch, all_data, train=True):
        total_loss = 0
        total_data = 0

        str_code = "train" if train else "test"
        data_itr = tqdm.tqdm(enumerate(all_data),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(all_data),
                              bar_format="{l_bar}{r_bar}")
        if not train:
            torch.set_grad_enabled(False)
        for idx, data in data_itr:
            self.optimizer.zero_grad()
            total_data+=1
            data = data.cuda()
            result = self.model(data)
            encoding_loss = self.criterion(result, data)
            total_loss+=float(encoding_loss)
            if train:
                encoding_loss.backward()
                self.optimizer.step()
        if not train: 
            torch.set_grad_enabled(True)
        return total_loss/total_data

    def save(self, epoch, file_path="output/autoencoder.model"):
        """
        Saving the current model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.state_dict(), output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


class ImageTrainer():
    def __init__(self, auto_enc: ImageAutoEncoder, dataloader_train, dataloader_test, l_rate):
        self.model = auto_enc
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(list(self.model.encoder.lin.parameters()) + list(self.model.decoder.lin.parameters()), lr=l_rate)
        self.train_data = dataloader_train
        self.test_data = dataloader_test

    def train(self, epoch):
        loss = self.iteration(epoch, self.train_data)
        return loss

    def test(self, epoch):
        loss = self.iteration(epoch, self.test_data, train=False)
        return loss

    def iteration(self, epoch, all_data, train=True):
        total_loss = 0
        total_data = 0

        str_code = "train" if train else "test"
        data_itr = tqdm.tqdm(enumerate(all_data),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(all_data),
                              bar_format="{l_bar}{r_bar}")
        for idx, data in data_itr:
            self.optimizer.zero_grad()
            total_data+=1
            data = data.cuda()
            result = self.model(data)
            encoding_loss = self.criterion(result, data)
            total_loss+=float(encoding_loss)
            if train:
                encoding_loss.backward()
                self.optimizer.step()
        return total_loss/total_data

    def save(self, epoch, file_path="output/autoencoder.model"):
        """
        Saving the current model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.state_dict(), output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path