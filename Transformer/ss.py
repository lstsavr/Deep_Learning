import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(326)

embedding_table = nn.Embedding(size,num_embedding)