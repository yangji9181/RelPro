from data import InputData
import numpy
from ntn_model import NTN_Model
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import pickle
from sklearn import svm
import random


class NTN():

    def __init__(self,
                 output_file_name,
                 emb_dimension=64,
                 batch_size=128,
                 iteration=1,
                 initial_lr=5e-3,
                 neg_sample_size = 3):
        self.output_file_name = output_file_name
        self.neg_sample_size = neg_sample_size
        self.data = InputData(batch_size)
        self.emb_size = self.data.node_size
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.tensor_slice = 64
        self.input_word_emb_dim = 100
        #note encoded size must agree with tensor slice in order for the dimension to match
        self.NTN_model = NTN_Model(self.emb_size, self.input_word_emb_dim,
                                   self.tensor_slice, self.emb_dimension, self.tensor_slice)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.NTN_model.cuda()
        self.optimizer = torch.optim.Adam(self.NTN_model.parameters(), lr=self.initial_lr)

        # self.optimizer = optim.SGD(self.NTN_model.parameters(), lr=self.initial_lr)

    def train(self, iteration):
        """Multiple training.
        Returns:
            None.
        """
        batch_count = iteration
        process_bar = tqdm(range(int(batch_count)))
        # self.skip_gram_model.save_embedding(
        #     self.data.id2word, 'begin_embedding.txt', self.use_cuda)
        for i in process_bar:
            e1, e2, ground_truth, title_emb = self.data.generate_batch_sampling(self.neg_sample_size)

            e1 = Variable(torch.LongTensor(e1))
            e2 = Variable(torch.LongTensor(e2))
            ground_truth = Variable(torch.FloatTensor(ground_truth))
            title_emb = Variable(torch.FloatTensor(title_emb))

            if self.use_cuda:
                e1 = e1.cuda()
                e2 = e2.cuda()
                ground_truth = ground_truth.cuda()
                title_emb = title_emb.cuda()
            self.optimizer.zero_grad()
            loss = self.NTN_model.forward(e1, e2, title_emb, ground_truth)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (loss.item() / self.batch_size,
                                         self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.NTN_model.save_embedding(self.output_file_name,  use_cuda=self.use_cuda)

if __name__ == '__main__':
    # output_file = 'author_emb_weighted.txt'
    # n2v = NTN(output_file_name=output_file)
    # n2v.train(500000)
    # #
    # emb_size = 5
    # output_file = 'author_emb_weighted_'+str(emb_size)+'.txt'
    # n2v = NTN(output_file_name=output_file, emb_dimension=emb_size)
    # n2v.train(50000)
    #
    # emb_size = 8
    # output_file = 'author_emb_weighted_'+str(emb_size)+'.txt'
    # n2v = NTN(output_file_name=output_file, emb_dimension=emb_size)
    # n2v.train(50000)
    #
    # emb_size = 16
    # output_file = 'author_emb_weighted_'+str(emb_size)+'.txt'
    # n2v = NTN(output_file_name=output_file, emb_dimension=emb_size)
    # n2v.train(50000)
    #
    # emb_size = 32
    # output_file = 'author_emb_weighted_'+str(emb_size)+'.txt'
    # n2v = NTN(output_file_name=output_file, emb_dimension=emb_size)
    # n2v.train(50000)

    emb_size = 64
    output_file = 'author_emb_weighted_'+str(emb_size)+'.txt'
    n2v = NTN(output_file_name=output_file, emb_dimension=emb_size)
    n2v.train(20000)