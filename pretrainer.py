
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

import tqdm

from Screen2Vec import Screen2Vec
from prediction import TracePredictor
from vocab import ScreenVocab

class Screen2VecTrainer:
    """
    Trains a Screen2Vec embedding using a prediction task
    """

    def __init__(self, predictor: TracePredictor, vocab_train: ScreenVocab, vocab_test: ScreenVocab, dataloader_train, dataloader_test, 
                l_rate: float, neg_samp: int):
        """
        predictor: TracePredictor module
        vocab_train: a ScreenVocab from which to find a negative sample for the training data
        vocab_test: a ScreenVocab from which to find a negative sample for the testing data
        dataloader_train, dataloader_test: dataloaders
        l_rate: learning rate for optimizer
        neg_samp: number of negative samples to compare against for training data
        """
        self.predictor = predictor 
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = Adam(self.predictor.parameters(), lr=l_rate)
        self.vocab_train = vocab_train
        self.vocab_test = vocab_test
        self.train_data = dataloader_train
        self.test_data = dataloader_test
        self.neg_sample_num = neg_samp

    def train(self, epoch):
        loss = self.iteration(epoch, self.train_data)
        return loss

    def test(self, epoch):
        loss = self.iteration(epoch, self.test_data, train=False)
        return loss

    def iteration(self, epoch, data_loader: iter, train=True):
        """
        loop over the data_loader for training or testing
        if train , backward operation is activated
        also auto save the model every epoch

        :param epoch: index of current epoch 
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        # iterate through data_loader
        total_loss = 0
        total_data = 0

        str_code = "train" if train else "test"
        data_itr = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        # to avoid memory leak
        if not train:
            torch.set_grad_enabled(False)

        for idx, data in data_itr:
            self.optimizer.zero_grad()

            # load data 
            UIs, descr, trace_screen_lengths, indices, layouts = data
            total_data+=len(UIs)
            # move to GPU
            UIs = UIs.cuda()
            descr = descr.cuda()
            trace_screen_lengths = trace_screen_lengths.cuda()
            if layouts is not None:
                layouts = layouts.cuda()
            # get negative samples to compare against
            if train:
                UIs_comp, comp_descr, comp_tsl, comp_layouts = self.vocab_train.negative_sample(self.neg_sample_num, indices)
            else:
                # smaller negative sample for test data because there's less of it
                UIs_comp, comp_descr, comp_tsl, comp_layouts = self.vocab_test.negative_sample(int(self.neg_sample_num/8), indices)
            # move to GPU
            UIs_comp = UIs_comp.cuda()
            comp_descr = comp_descr.cuda()
            comp_tsl = comp_tsl.cuda()
            if comp_layouts is not None:
                comp_layouts = comp_layouts.cuda()

            # forward the prediction models
            c, result, context = self.predictor(UIs, descr, trace_screen_lengths, layouts) #input here
            h_comp = self.predictor.model(UIs_comp, comp_descr, comp_tsl, comp_layouts, False).squeeze(0)

            # dot products to find out similarity
            # with negative sampling
            neg_dot_products = torch.mm(c, h_comp.transpose(0,1).cuda())
            # with other screens in trace
            neg_self_dot_products = torch.bmm(c.unsqueeze(1), context.transpose(1,2)).squeeze(1)
            # with targets
            pos_dot_products = torch.mm(c, result.transpose(0,1).cuda())
            correct = torch.from_numpy(np.arange(0,len(UIs)))
            dot_products = torch.cat((pos_dot_products, neg_dot_products, neg_self_dot_products), dim=1)
            dot_products = dot_products.cpu()

            # calculate loss for this batch
            prediction_loss = self.criterion(dot_products, correct.long())
            total_loss+=float(prediction_loss)

            # if in train, backwards and optimization
            if train:
                prediction_loss.backward()
                self.optimizer.step()
        if not train: 
            torch.set_grad_enabled(True)
        return total_loss/total_data
        

    def save(self, epoch, file_path="output/trained.model"):
        """
        Saving the current model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.predictor.state_dict(), output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path