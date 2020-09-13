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
from autoencoder import ScreenVisualLayout, ScreenVisualLayoutDataset, ImageAutoEncoder, ImageTrainer
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True, type=str, help="dataset of screens to train on")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="traces in a batch")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-r", "--rate", type=float, default=0.001, help="learning rate")
parser.add_argument("-t", "--type", type=int, default=0, help="0 to create layout autoencoder, 1 to create visual autoencoder")
parser.add_argument("-m", "--model", type=str, default="", help="path to layout autoencoder if training a visual one")

args = parser.parse_args()

if args.type == 0:
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
        print(test_loss)
        print("--------")

        if (epoch%50)==0:
            print("saved on epoch " + str(epoch))
            trainer.save(epoch)
    plot_loss(train_loss_data, test_loss_data, "output/autoencoder")
    trainer.save(args.epochs, "output/autoencoder")
        

elif args.type == 1:
    dataset = ScreenVisualLayoutDataset(args.dataset)

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


    model = ImageAutoEncoder()
    layout_model = LayoutAutoEncoder()
    layout_model.load_state_dict(torch.load(args.model))
    model.encoder.layout_encoder = layout_model.enc
    model.decoder.layout_decoder = layout_model.dec
    model.cuda()

    trainer = ImageTrainer(model, train_loader, test_loader, args.rate)

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
        print(test_loss)
        print("--------")

        if (epoch%50)==0:
            print("saved on epoch " + str(epoch))
            trainer.save(epoch, "output/visual_encoder_fast")
    plot_loss(train_loss_data, test_loss_data, "output/visual_encoder_fast")
    trainer.save(args.epochs, "output/visual_encoder_fast")



