import argparse
import numpy as np
import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from UI_embedding.plotter import plot_loss
from autoencoder import ScreenLayoutDataset, LayoutAutoEncoder, LayoutTrainer

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True, type=str, help="dataset of screens to train on")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="traces in a batch")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-r", "--rate", type=float, default=0.001, help="learning rate")

args = parser.parse_args()





dataset = ScreenLayoutDataset(args.dataset)

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)


model = LayoutAutoEncoder()
model.cuda()

trainer = LayoutTrainer(model, train_loader, test_loader, args.rate)

train_loss_data = []
test_loss_data = []
for epoch in tqdm.tqdm(range(args.epochs)):
    print("--------")
    print(str(epoch) + " loss:")
    train_loss = trainer.train(epoch)
    print(train_loss)
    print("--------")
    train_loss_data.append(train_loss)

    test_loss = trainer.test(epoch)
    test_loss_data.append(test_loss)

    if (epoch%50)==0:
        print("saved on epoch " + str(epoch))
        trainer.save(epoch)
plot_loss(train_loss_data, test_loss_data, "output/autoencoder")
trainer.save(args.epochs, "output/autoencoder")
    


