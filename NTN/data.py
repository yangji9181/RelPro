import pickle
import numpy as np



class InputData():
    def __init__(self, batch_size, input_file_name):
        self.read_input_data(input_file_name)
        self.node_ids, self.rel_ids = self.get_attribute_rel_id()
        self.batch_size = batch_size
        self.attr_size = len(self.node_ids)
        self.rel_size = len(self.rel_ids)

    def read_input_data(self, input_file_name):
        '''
        :param input_file_name: input file name
        :return: training data in numpy array in form of [[id1, id2, rel_id],....]
        '''
        temp = open(input_file_name, 'r').readlines()[1:]
        self.train_data = []
        for i in range(len(temp)):
            self.train_data.append(temp[i].strip('\n').split(','))
        self.train_data = np.array(self.train_data).astype(int)


    def get_attribute_rel_id(self):
        '''
        :return: return unique node id and relation id
        '''
        node_ids = {}
        rel_ids = {}
        for i in range(len(self.train_data)):
            node_ids[self.train_data[i][0]] = 1
            node_ids[self.train_data[i][1]] = 1
            rel_ids[self.train_data[i][2]] = 1
        return list(node_ids.keys()), list(rel_ids.keys())

    def generate_pos_sampling(self):
        cur_batch = self.train_data[np.random.choice(len(self.train_data), self.batch_size)]
        return cur_batch
    #now neg sampling might get the neighbors, if performance bad then modify this sampling
    def generate_neg_sampling(self, neg_num):
        neg_v = np.random.choice(self.node_ids, size=self.batch_size * neg_num).tolist()
        return neg_v

    def generate_batch_sampling(self, neg_sample):
        '''

        :param neg_sample: number of neg sample
        :return: id1, id2, relation id, grond_truth meaning whether id1, id2, formed by relation id
        '''
        pos_batch = self.generate_pos_sampling()
        pos_batch = np.concatenate((pos_batch, np.ones_like(pos_batch[:,0].reshape(-1,1))), axis=1)
        neg_batch = np.repeat(pos_batch, neg_sample, axis=0)
        neg_id = self.generate_neg_sampling(neg_sample)
        neg_batch[:, 3] = 0
        neg_batch[:, 0] = neg_id[:]
        # neg_batch[1::2, 1] = neg_id[1::2]
        batch_data = np.concatenate((pos_batch, neg_batch), axis=0)
        e1 = batch_data[:, 0].astype(int)
        e2 = batch_data[:, 1].astype(int)
        rel_ids = batch_data[:, 2].astype(int)
        ground_truth = batch_data[:, 3].astype(int)

        return e1, e2, rel_ids, ground_truth


