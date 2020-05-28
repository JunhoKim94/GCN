import numpy as np
import scipy.sparse as sp
import torch

def encode_onehot(labels):
    '''
    labels (data_num, class)
    '''
    word2idx = dict()
    for word in labels:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    onehot = np.eye(len(word2idx))

    return np.array([onehot[word2idx[word]] for word in labels])
    
    
def load_data(path, dataset):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype = np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype = np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    idx = np.array(idx_features_labels[:, 0], dtype = np.int32)


    #graph
    idx = {j : i for i, j in enumerate(idx_features_labels[:,0])}


if __name__ == "__main__":
    path = "./data/cora/"
    dataset = "cora"
    load_data(path, dataset)