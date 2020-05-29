import torch
import torch.nn as nn
import numpy as np
from preprocess import load_data


adj, features, labels, idx_train, idx_val, idx_test = load_data("./data", "cora")