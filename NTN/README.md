## NTN Model 
This folder contains the implementation of the NTN Model. The code implemented the main model mentioned in the [Reasoning With Neural Tensor Networks for Knowledge Base Completion](https://cs.stanford.edu/~danqi/papers/nips2013.pdf).

## Data input format and Output result
The input for model is tuple in the format of (id1, id2, rel_id, ground truth).
* Id1, Id2 is the id number for the entity. After training, the model will learn an embedding representation on all the entity from the dataset.
* rel_id is the id number for the relation that connect id1, id2 in the above example. After training, the model will learn an embedding representation for all the relations from the dataset.
* Ground truth is either 0 or 1, representing whether id1 and id2 are connected by some relation rel_id. When ground truth is 1, it represents id1 and id2 are connected by rel_id. Vise Versa. The ground truth 0 are negative examples generated randomly to train model.
* An example:
    * (author1, author2, collaboration relation, 1), (author1, author3, collaboration relation, 0) Assume that author1 and author2 have collaboration relationship in social network, whereas author1, author3 doesn't have any collaboration.
* An example datafile `example_training_data.txt` is provided. In the file, it's in the format of id1, id2, rel_id, where groud truth in the file is assumed to be 1 for all the example. In order to run this file, `data.py` has to be used in order to generate negative example samples for model to learn. 
* Have a look at the `generate_batch_sampling(num_neg)` function in `data.py` for detailed negative sample generation.

## Custom Data

In order to use your own data, you have to provide 
* file that has data format of (id1, id2, rel_id) which only contains the positive example.

