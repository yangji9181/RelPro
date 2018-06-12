import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pickle



class NTN_Model(nn.Module):

    @staticmethod
    def mse_loss(input, target):
        num = input.data.nelement()
        return torch.sum((input - target) ** 2)


    def __init__(self, input_size, word_emb_size, encoded_size, entity_emb_size, tensor_slice):
        '''

        :param input_size: the total number of word tokens in the corpus
        :param word_emb_size: the input word embedding size for the relation decoder
        :param encoded_size: the size of the encoded embedding size for the relation/attribute
        :param entity_emb_size: the learned embedding dimension of all entities
        :param tensor_slice: The number of slice in the bilinear module.
        The bilinear module has size entity_emb_size* entity_emb_size * tensor_slice
        '''


        super(NTN_Model, self).__init__()
        #below is for encoder layers,
        #H is the middle layer for encoder
        self.bilinear = torch.nn.Bilinear(entity_emb_size, entity_emb_size, tensor_slice)
        self.linear = torch.nn.Linear(2 * entity_emb_size, tensor_slice)

        self.weighted = torch.nn.Linear(encoded_size, encoded_size)
        self.weighted_sum = nn.Parameter(torch.rand(encoded_size)* -2 + 1)
        self.H = 200
        self.encoder_r = torch.nn.Sequential(
            torch.nn.Linear(word_emb_size, self.H),
            torch.nn.ReLU(),
            torch.nn.Linear(self.H, encoded_size))

        self.decoder_r = torch.nn.Sequential(
            torch.nn.Linear(encoded_size, self.H),
            torch.nn.ReLU(),
            torch.nn.Linear(self.H, word_emb_size),
            nn.Sigmoid())


        self.embedding_size = input_size
        self.embedding_dim = entity_emb_size
        self.embeddings = nn.Embedding(self.embedding_size, self.embedding_dim)
        self.init_emb()


    def init_emb(self):
        '''
        Used initlized embedding for each of entity Here the bilinear module and linear are initlized to be xavier_uniform
        :return:
        '''
        self.embeddings.weight.data.normal_(0, 1.0 / math.sqrt(self.embedding_size))
        nn.init.xavier_uniform_(self.bilinear.weight.data)
        nn.init.xavier_uniform_(self.linear.weight.data)
        # self.embeddings.weight.data.unfiform(-1, 1)

    def forward(self, e1, e2, title_emb, ground_truth):
        '''

        :param attr1: the word embedding for attribute 1 for author or (conf) other node type
        :param attr2: the word emebdding for attribute 2 for author or (conf) other node type
        :param rel: the word embedding for relationship that connect attribute 1 and attribute 2(paper abstract, paper title)
        :param ground_truth: the ground truth whether these two attributes are connected through this relation
        :return: the loss value
        '''
        eps = 1e-16 #add for numerical stability
        e1_emb = self.embeddings(e1)
        e2_emb = self.embeddings(e2)
        bilinear_out = self.bilinear(e1_emb, e2_emb)
        linear_out = self.linear(torch.cat((e1_emb, e2_emb) , 1))
        ntn_output = F.tanh(bilinear_out + linear_out)


        encoded_rel = self.encoder_r(title_emb)
        decoded_rel = self.decoder_r(encoded_rel)
        # weighted_encoded_rel = self.weighted_sum(encoded_rel)
        pred = F.sigmoid(torch.sum(encoded_rel *self.weighted_sum* ntn_output, dim=1))

        loss = -1 * (ground_truth * torch.log(pred + eps) + (1 - ground_truth) * torch.log(1 - pred + eps))
        loss1 = torch.sum(loss)
        loss2 = (self.mse_loss(title_emb, decoded_rel))
        return loss1 + 0.005 * loss2

    def save_embedding(self, file_name, use_cuda):

        """
        Save all embeddings to file.
        As this class only record word id, so the map from id to word has to be transfered from outside.
        :param file_name: file name.
        :param use_cuda: whether cuda is used when training
        :return None

        """
        if use_cuda:
            embedding = self.embeddings.weight.cpu().data.numpy()

        else:
            embedding = self.embeddings.weight.data.numpy()
        fout = open(file_name, 'wb')
        '''
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))
        '''
        pickle.dump(embedding, fout)
        fout.close()


    # def forward_emb(self, attr1, attr2, rel, ground_truth):
    #     eps = 1e-16
    #     emb_attr1 = self.embeddings(attr1)
    #     emb_attr2 = self.embeddings(attr2)
    #     encoded_rel = self.encoder_r(rel)
    #     pred = F.sigmoid(torch.sum(emb_attr1 * emb_attr2 * encoded_rel, dim=1))
    #     loss = -1 * (ground_truth * torch.log(pred + eps) + (1 - ground_truth) * torch.log(1 - pred + eps))
    #     return torch.sum(loss)