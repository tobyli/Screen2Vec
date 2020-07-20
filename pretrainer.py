
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import tqdm

from Screen2Vec import Screen2Vec
from prediction import TracePredictor
from vocab import ScreenVocab

class Screen2VecTrainer:
    """
    """

    def __init__(self, predictor: TracePredictor, vocab_train: ScreenVocab, vocab_test: ScreenVocab, dataloader_train, dataloader_test, 
                l_rate: float, neg_samp: int):
        """
        """
        self.predictor = predictor 
        self.criterion = nn.CrossEntropyLoss()
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
        total_batches = 0

        str_code = "train" if train else "test"
        data_itr = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        if not train:
            torch.set_grad_enabled(False)
        for idx, data in data_itr:
            total_batches+=1

            # load data properly
            UIs, descr, trace_screen_lengths, indices = data
            UIs = UIs.cuda()
            descr = descr.cuda()
            trace_screen_lengths = trace_screen_lengths.cuda()
            if train:
                UIs_comp, comp_descr, comp_tsl = self.vocab_train.negative_sample(self.neg_sample_num, indices)
            else:
                UIs_comp, comp_descr, comp_tsl = self.vocab_test.negative_sample(int(self.neg_sample_num/8), indices)
            UIs_comp = UIs_comp.cuda()
            comp_descr = comp_descr.cuda()
            comp_tsl = comp_tsl.cuda()
            # forward the training stuff (prediction models)
            c,result = self.predictor(UIs, descr, trace_screen_lengths) #input here
            h_comp = self.predictor.model(UIs_comp, comp_descr, comp_tsl).squeeze(0)
            

            neg_dot_products = torch.mm(c, h_comp.transpose(0,1).cuda())
            pos_dot_products = torch.bmm(c.unsqueeze(1), result.unsqueeze(2).cuda()).squeeze(-1)
            # calculate NLL loss for all prediction stuff
            dot_products = torch.cat((pos_dot_products, neg_dot_products), dim=1)
            dot_products = dot_products.cpu()
            prediction_loss = self.criterion(dot_products, torch.zeros(len(UIs)).long())
            total_loss+=float(prediction_loss)
            # if in train, backwards and optimization
            if train:
                prediction_loss.backward()
                self.optimizer.step()
            torch.cuda.empty_cache()
        if not train: 
            torch.set_grad_enabled(True)
        return total_loss/total_batches
        

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