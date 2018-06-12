# RelPro

This is currently on going project, the files contains in the folder will need modification in order to apapt to current project.

## Requirements
* pytorch (>=0.31)
* numpy
* tqdm

## Data

Current dataset is [DBLP-citation-network-V10](https://aminer.org/citation).
Data need to be preprocessed prior to using it. The preprossing tool is under the folder name preprocessing. The detailed documentation is in the folder.

## Models

The currently implemented models are 
* `NTN`:[Reasoning With Neural Tensor Networks for Knowledge Base Completion](https://cs.stanford.edu/~danqi/papers/nips2013.pdf)
Please refer to `ntn_model` for detailed implementation.
* `HIN2VEC`:[HIN2Vec: Explore Meta-paths in Heterogeneous Information Networks for Representation Learning](https://dl.acm.org/citation.cfm?id=3132953)
Please refer to `hi2vec_model.py` for detailed implementation.


