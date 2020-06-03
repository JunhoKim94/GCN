import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import GraphConv, GAT_layer, MultiHeadGAT
from dgl.nn.pytorch import GATConv

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

class GAT(nn.Module):
    def __init__(self, feature, hidden, output, num_head, dropout):
        super(GAT, self).__init__()

        self.gt1 = MultiHeadGAT(feature, hidden, num_head, dropout, bias = True,activation = F.leaky_relu)
        self.gt2 = MultiHeadGAT(hidden * num_head, output, num_head, dropout, bias = True, activation= F.leaky_relu)

        
    def forward(self, x, adj):
        
        output = self.gt1(x, adj)
        output = self.gt2(output, adj)

        return output

