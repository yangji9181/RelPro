import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pickle

class Hin2Vec_model(nn.Module):

    def __init__(self, input_data_size, relation_size, embedding_dim):
        '''

        :param input_data_size:number of node in the graph, node has to be same type
        :param relation_size:number of relations(meta-path) in the graph
        :param embedding_dim:embedding dimension
        '''


        super(Hin2Vec_model, self).__init__()

        self.embedding_dim = embedding_dim
        self.embedding_size = input_data_size
        self.relation_size = relation_size
        self.embeddings = nn.Embedding(self.embedding_size, self.embedding_dim)
        self.relation_embedding = nn.Embedding(self.relation_size, self.embedding_dim)
        self.init_emb()


    def init_emb(self):
        '''
        Initalize init embedding
        '''
        self.embeddings.weight.data.normal_(0, 1.0 / math.sqrt(self.embedding_size))
        self.relation_embedding.weight.data.normal_(0, 1.0 / math.sqrt(self.embedding_size))
        # self.embeddings.weight.data.unfiform(-1, 1)



    def forward(self, attr1, attr2, rel, ground_truth):
        '''
        @todo use a different function that map values to range 0~1, currently the mapping function is sigmoid
        :param attr1: list node 1 id, here attribute 1 must be the same type as attribute 2
        :param attr2: list of node id co-occur with node 1 id
        :param rel: list of relation id, here realtion id could be the meta-path id
        :param ground_truth: ground truth for whether attr1 and attr2 are connect by some relation
        :return: the loss value
        '''
        eps = 1e-16
        emb_attr1 = self.embeddings(attr1)
        emb_attr2 = self.embeddings(attr2)
        emb_rel = self.relation_embedding(rel)
        sig_encoded_rel = F.sigmoid(emb_rel)
        pred = F.sigmoid(torch.sum(emb_attr1 * emb_attr2 * sig_encoded_rel, dim=1))
        loss = -1 * (ground_truth * torch.log(pred + eps) + (1 - ground_truth) * torch.log(1 - pred + eps))
        loss1 = torch.sum(loss)

        return loss1

    def save_embedding(self, file_name, use_cuda):
        '''
        :param file_name:the output file name to save the *attribute* embedding,
        :param use_cuda: whether cuda is used to run the code.
        :return: None
        '''
        if use_cuda:
            embedding = self.embeddings.weight.cpu().data.numpy()

        else:
            embedding = self.embeddings.weight.data.numpy()
        fout = open(file_name, 'wb')
        pickle.dump(embedding, fout)
        fout.close()




def test_look_up_emb():
    fout = open('example_training_data.txt', 'w')
    model = Hin2Vec_model(200, 5, 32)
    dtype = torch.FloatTensor
    attr1_id_np = np.random.randint(0,199,size=(2000,))
    attr1_id = Variable(torch.LongTensor(attr1_id_np))
    attr2_id_np = np.random.randint(0,199,size=(2000,))
    attr2_id = Variable(torch.LongTensor(attr2_id_np))
    rel_id_np = np.random.randint(0, 5, size=(2000,))
    rel_id = Variable(torch.LongTensor(rel_id_np))
    gt = torch.rand(2000) > 0.4
    for i in range(2000):
        fout.write(str(attr1_id_np[i]) + ',' + str(attr2_id_np[i]) + ',' + str(rel_id_np[i]) + '\n')
    ground_truth = Variable(gt.type(dtype), requires_grad=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        attr1_id = attr1_id.cuda()
        attr2_id = attr2_id.cuda()
        rel_id = rel_id.cuda()
        ground_truth = ground_truth.cuda()
    for j in range(0,1000):
        for i in range(int(2000/50)):
            optimizer.zero_grad()
            loss = model.forward(attr1_id[5*i:5*i+5], attr2_id[5*i:5*i+5], rel_id[5*i:5*i+5], ground_truth[5*i:5*i+5])
            if i % 10 == 0:
                print(i, float(loss))
            loss.backward()
            optimizer.step()






if __name__ == '__main__':
    test_look_up_emb()