import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import GraphConv

class GCN(nn.Module):
    def __init__(self, feature, hidden, output, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(feature, hidden, dropout)
        self.gc2 = GraphConv(hidden, output, dropout, activation = F.log_softmax)


    def forward(self, x, adj):
        #(n, hidden), (n,n)
        output, adj = self.gc1(x, adj)
        #(n,class), (n,n)
        output, adj = self.gc2(output, adj)

        #n, class
        return output