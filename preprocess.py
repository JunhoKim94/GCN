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

def normalize(mx):
    '''
    row normalize sparse matrix
    '''
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx

def sparse_to_torch(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack([sparse_mx.row, sparse_mx.col]).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)

    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    
def load_data(path, dataset):
    '''
    path = datasets folder path(ex) ./data)
    dataset = sepecific dataset name(ex) cora)
    '''

    #network dataset
    idx_features_labels = np.genfromtxt("{}/{}.content".format(path, dataset), dtype = np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype = np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    idx = np.array(idx_features_labels[:, 0], dtype = np.int32)

    #graph
    idx = {j : i for i, j in enumerate(idx_features_labels[:,0].astype(np.int32))}
    edges = np.genfromtxt("{}/{}.cites".format(path, dataset), dtype = np.int32)
    #change node idx to mine
    edges = np.array(list(map(idx.get, edges.flatten())), dtype = np.int32).reshape(edges.shape)
    #adj
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0] , edges[:, 1])), shape = (labels.shape[0], labels.shape[0]))

    #build symmetric adj
    adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    print(adj)
    #print(idx.get)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = torch.LongTensor(range(140))
    idx_val = torch.LongTensor(range(200, 500))
    idx_test = torch.LongTensor(range(500, 1000))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_to_torch(adj)

    return adj, features, labels, idx_train, idx_val, idx_test

if __name__ == "__main__":
    path = "./data/cora"
    dataset = "cora"
    load_data(path, dataset)