import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConv(nn.Module):

    def __init__(self, inputs, outputs, dropout, bias = True, activation = F.relu):
        super(GraphConv, self).__init__()

        self.dropout = dropout
        self.inputs = inputs
        self.outputs = outputs
        self.act = activation

        self.weight = nn.Parameter(torch.FloatTensor(inputs, outputs))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(outputs))
        
        self.initialize()
        
    def initialize(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        '''
        x : node feature (n , f) --> dense matrix
        adj : adjacency matrix (n, n) --> sparse matrix

        return activation(adj * x * W)
        '''

        x =  F.dropout(x, p = self.dropout)
        out = torch.mm(x, self.weight)
        
        if self.bias is not None:
            out = torch.add(out, self.bias)

        out = torch.spmm(adj, out)

        return self.act(out), adj
        

class GAT_layer(nn.Module):
    def __init__(self, inputs, outputs, dropout, bias = True , activation = F.leaky_relu):
        super(GAT_layer, self).__init__()

        self.dropout = dropout
        self.inputs = inputs
        self.outputs = outputs
        self.act = activation

        self.weight = nn.Parameter(torch.FloatTensor(inputs, outputs))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(outputs))
        
        self.att = nn.Parameter(torch.FloatTensor(2 * outputs, 1))

        self.initialize()
        
    def initialize(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        '''
        x : feature (n, f)
        '''
        n = x.shape[0]

        x = F.dropout(x, p = self.dropout)
        #(n, h)
        out = torch.mm(x, self.weight)
        #(n * n,h_i) + (n , h_0 ~ h_n)
        output = torch.cat([out.repeat(1, n).view(n * n, -1), out.repeat(n,1)]).view(n, -1 , 2 * self.outputs)
        #print(output.shape, out.shape, self.att.shape)
        att = self.act(torch.matmul(output, self.att).squeeze(2))
        #print(att.shape)

        #adj = adj.to_dense()

        zeros = -1e15 * torch.ones_like(adj)
        att = torch.where(adj > 0, att, zeros)

        att = F.softmax(att)

        return F.sigmoid(torch.mm(att, out))

class MultiHeadGAT(nn.Module):
    def __init__(self, inputs, outputs, num_head, dropout, bias, activation, merge = 'cat'):
        super(MultiHeadGAT, self).__init__()

        self.heads = num_head

        self.multi = nn.ModuleList([GAT_layer(inputs, outputs, dropout, bias, activation) for _ in range(num_head)])

        self.merge = merge

    def forward(self, h, adj):
        out = [att_head(h, adj) for att_head in self.multi]

        if self.merge.lower() == 'cat':
            return torch.cat(out, dim = 1)

        else:
            return torch.mean(torch.stack(out))