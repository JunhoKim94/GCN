import torch
import torch.nn as nn
import numpy as np
from preprocess import load_data
from model.model import GCN, GAT

#initialize 안하면 수렴 안댐.
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
print(f"GPU Usable : {torch.cuda.is_available()}")

adj, features, labels, idx_train, idx_val, idx_test = load_data("./data/cora", "cora")

#adj = adj.to_dense()
adj.to(device)
features.to(device)
labels.to(device)
idx_train.to(device)
idx_val.to(device)
idx_test.to(device)

print(features.shape[1], labels.max().item())

feat = features.shape[1]
hidden = 16
output = 7
dropout = 0.2
lr = 0.001
epochs = 1000
w_decay = 5e-4

model = GCN(feat, hidden, output, dropout)
#model = GAT(feat, hidden, output, 8, dropout)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = w_decay)

model.to(device)

def accuracy(output, labels):
    preds = output.max(1)[1]
    #print(preds.shape)
    score = preds[preds == labels]
    score = len(score) / len(labels)
    
    return score
    

total_loss = 0
for epoch in range(1, epochs + 1):
    
    output = model(features, adj)

    loss = criterion(output[idx_train], labels[idx_train])

    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()

    total_loss = loss.item()


    if epoch % 50 == 0:
        train_score = accuracy(output[idx_train], labels[idx_train])
        val_score = accuracy(output[idx_val], labels[idx_val])
        test_score = accuracy(output[idx_test], labels[idx_test])

        print(f"epoch : {epoch}  |  loss : {total_loss}  | score : {train_score} | val : {val_score} | test : {test_score}")
