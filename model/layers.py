import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphConv(nn.Module):

    def __init__(self, inputs, outputs, dropout, bias, activation = nn.ReLU):
        super(GraphConv, self).__init__()

        self.dropout = dropout
        self.inputs = inputs
        self.outputs = outputs
        self.act = activation

        self.weight = nn.Parameter(torch.FloatTensor(inputs, outputs))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(outputs))
        
    def initialize(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        '''
        x : node feature (n , f)
        adj : adjacency matrix (n, n)
        '''

        x =  F.dropout(x, p = self.dropout)

        output = torch.mm()
        